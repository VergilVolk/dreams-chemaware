#!/usr/bin/env python
"""Export frozen ChemAware V2 actions into grand-fusion score contracts.

Two mutually exclusive modes are supported:

* ``--evidence`` checks alignment against the complete MassSpecGym pair
  evidence and writes an expert NPZ accepted by
  ``extend_grand_fusion_pair_evidence.py``.
* ``--benchmark`` scores both sealed GNPS panels and writes an immutable
  ``gnps_gold_silver_10ppm_pair_score_cache_v1`` directory.

Inference sees spectra, candidate partitions and frozen scores only.  Labels,
true identities and correctness outcomes are never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries_truthblind
from chemaware_truthblind_candidate_core import predict_truthblind_policy
from gnps_pair_score_cache import PANELS, benchmark_fingerprint, sha256_file, write_pair_score_cache
from noise_final_core import sha256_file as noise_sha256_file


ROOT = Path(__file__).resolve().parents[1]
def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument(
        "--rule-library", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--evidence", type=Path)
    mode.add_argument("--benchmark", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-queries", type=int, default=256)
    parser.add_argument(
        "--expected-policy-sha256", default="",
        help="Optional second lock in addition to the SHA256 recorded in policy report.json",
    )
    return parser.parse_args()


def _kernel_args(token_dir: Path, rule_library: Path, replay: dict) -> SimpleNamespace:
    return SimpleNamespace(
        token_dir=token_dir,
        rule_library=rule_library,
        top_peaks=int(replay["top_peaks"]),
        kernel_dim=int(replay["kernel_dim"]),
        bin_width=float(replay["bin_width"]),
        grid_offsets=int(replay["grid_offsets"]),
        intensity_power=float(replay["intensity_power"]),
        mass_shift_da=float(replay["mass_shift_da"]),
        pair_weight=float(replay["pair_weight"]),
        multi_bin_widths=tuple(replay["multi_bin_widths"]),
        uniform_channel_weight=float(replay["uniform_channel_weight"]),
        rule_tolerance=float(replay["rule_tolerance"]),
        rule_channel_weight=float(replay["rule_channel_weight"]),
    )


def _load_policy(args: argparse.Namespace) -> tuple[dict, dict, str]:
    policy_path = args.policy_dir / "truthblind_policy.joblib"
    report_path = args.policy_dir / "report.json"
    for path in (policy_path, report_path, args.rule_library):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    observed = sha256_file(policy_path)
    declared = report.get("frozen_truthblind_policy", {}).get("sha256")
    if not declared or observed != declared:
        raise RuntimeError(f"ChemAware policy/report SHA256 mismatch: {observed} != {declared}")
    if args.expected_policy_sha256 and observed != args.expected_policy_sha256:
        raise RuntimeError(
            f"ChemAware policy hash drifted: {observed} != {args.expected_policy_sha256}"
        )
    bundle = joblib.load(policy_path)
    if bundle.get("schema") != "chemaware_truthblind_candidate_policy_v1":
        raise RuntimeError("unsupported ChemAware policy schema")
    replay = report.get("replay_contract", {}).get("arguments")
    if not isinstance(replay, dict):
        raise RuntimeError("policy report does not contain the frozen replay contract")
    return bundle, replay, observed


def _token_state(token_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    required = (
        "rows.npy", "official_embeddings_f32.npy", "mz_f32.npy",
        "intensity_f32.npy", "valid.npy", "precursor_mz_f32.npy",
    )
    for name in required:
        if not (token_dir / name).is_file():
            raise FileNotFoundError(token_dir / name)
    rows = np.load(token_dir / "rows.npy").astype(np.int64)
    official = np.load(token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    if official.ndim != 2 or len(official) != len(rows) or len(np.unique(rows)) != len(rows):
        raise RuntimeError("ChemAware token row/embedding cache is malformed")
    return rows, official, {int(row): index for index, row in enumerate(rows)}


def _body_from_panel(path: Path) -> dict[str, np.ndarray]:
    # Deliberately project only deployment-visible arrays; molecule_label and
    # identities remain unopened until the downstream evaluator.
    with np.load(path, allow_pickle=False) as panel:
        return {
            "query_row": np.asarray(panel["query_row"], dtype=np.int64),
            "query_ptr": np.asarray(panel["query_ptr"], dtype=np.int64),
            "molecule_ptr": np.asarray(panel["molecule_ptr"], dtype=np.int64),
            "pair_candidate_row": np.asarray(panel["candidate_row"], dtype=np.int64),
        }


def _promote_frozen_actions(
    body: dict[str, np.ndarray], policy: dict, cache: KernelCache,
    official: np.ndarray, row_position: dict[int, int], variants: tuple[str, ...],
    batch_queries: int, expected_official: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    query_count = len(body["query_row"])
    edge_count = len(body["pair_candidate_row"])
    output = np.empty(edge_count, dtype=np.float32)
    selected_total = 0
    abstained_total = 0
    for start in range(0, query_count, batch_queries):
        stop = min(query_count, start + batch_queries)
        queries = np.arange(start, stop, dtype=np.int64)
        scored = score_queries_truthblind(
            queries, body, official, row_position, cache, variants,
        )
        predictions = predict_truthblind_policy(
            policy, scored, [scored for _ in policy["control_rule_keys"]],
        )
        for local, query in enumerate(range(start, stop)):
            molecule_left = int(body["query_ptr"][query])
            molecule_right = int(body["query_ptr"][query + 1])
            edge_left = int(body["molecule_ptr"][molecule_left])
            edge_right = int(body["molecule_ptr"][molecule_right])
            values = np.asarray(scored["global"][local], dtype=np.float32).copy()
            if expected_official is not None and not np.allclose(
                values, expected_official[edge_left:edge_right], rtol=2e-5, atol=2e-6,
            ):
                raise RuntimeError(f"official score alignment drifted at query {query}")
            if not bool(predictions["abstained"][local]):
                selected = int(predictions["selected_candidate"][local])
                if not (0 <= selected < molecule_right - molecule_left):
                    raise RuntimeError(f"selected candidate is out of range at query {query}")
                pointer = np.asarray(scored["reference_ptr"][local], dtype=np.int64)
                left, right = int(pointer[selected]), int(pointer[selected + 1])
                best = left + int(np.argmax(values[left:right]))
                values[best] = np.nextafter(np.max(values), np.float32(np.inf))
                selected_total += 1
            else:
                abstained_total += 1
            output[edge_left:edge_right] = values
        print(f"[ChemAware V2] {stop:,}/{query_count:,} queries", flush=True)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("ChemAware exported scores are non-finite")
    return output, {"selected": selected_total, "abstained": abstained_total}


def _massspecgym(args: argparse.Namespace, policy: dict, replay: dict, policy_hash: str) -> None:
    if args.manifest is None:
        raise RuntimeError("--manifest is required with --evidence")
    evidence_path = args.evidence / "evidence.npz"
    evidence_report = json.loads((args.evidence / "report.json").read_text(encoding="utf-8"))
    if evidence_report.get("status") != "NOISE_MSG_PAIR_EVIDENCE_COMPLETE":
        raise RuntimeError("MassSpecGym evidence is not frozen")
    with np.load(evidence_path, allow_pickle=False) as evidence:
        query_ptr = np.asarray(evidence["query_ptr"], dtype=np.int64)
        molecule_ptr = np.asarray(evidence["molecule_ptr"], dtype=np.int64)
        formulas = np.asarray(evidence["query_formula"], dtype=str)
        expected_official = np.asarray(evidence["official_cosine"], dtype=np.float32)
    with np.load(args.manifest, allow_pickle=False) as manifest:
        body = {
            "query_row": np.asarray(manifest["query_row"], dtype=np.int64),
            "query_ptr": np.asarray(manifest["query_ptr"], dtype=np.int64),
            "molecule_ptr": np.asarray(manifest["molecule_ptr"], dtype=np.int64),
            "pair_candidate_row": np.asarray(manifest["pair_candidate_row"], dtype=np.int64),
        }
        manifest_formula = np.asarray(manifest["query_formula"], dtype=str)
    if not (
        np.array_equal(query_ptr, body["query_ptr"])
        and np.array_equal(molecule_ptr, body["molecule_ptr"])
        and np.array_equal(formulas, manifest_formula)
        and len(expected_official) == len(body["pair_candidate_row"])
    ):
        raise RuntimeError("ChemAware manifest does not align with frozen pair evidence")
    rows, official, row_position = _token_state(args.token_dir)
    variants = tuple(dict.fromkeys((
        "mass", str(policy["rule_key"]), *map(str, policy["control_rule_keys"]),
    )))
    cache = KernelCache(_kernel_args(args.token_dir, args.rule_library, replay), row_position, variants)
    scores, counts = _promote_frozen_actions(
        body, policy, cache, official, row_position, variants,
        args.batch_queries, expected_official,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    fd, raw = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".npz", dir=args.output.parent)
    os.close(fd)
    temporary = Path(raw)
    try:
        np.savez_compressed(
            temporary,
            pair_scores=scores,
            evidence_sha256=np.asarray(noise_sha256_file(evidence_path)),
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "status": "CHEMAWARE_V2_MASSSPECGYM_SCORE_EXPORT_COMPLETE",
        "labels_opened": False,
        "queries": int(len(body["query_row"])),
        "pairs": int(len(scores)),
        **counts,
        "policy_sha256": policy_hash,
        "manifest_sha256": sha256_file(args.manifest),
        "evidence_sha256": noise_sha256_file(evidence_path),
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def _gnps(args: argparse.Namespace, policy: dict, replay: dict, policy_hash: str) -> None:
    rows, official, row_position = _token_state(args.token_dir)
    variants = tuple(dict.fromkeys((
        "mass", str(policy["rule_key"]), *map(str, policy["control_rule_keys"]),
    )))
    cache = KernelCache(_kernel_args(args.token_dir, args.rule_library, replay), row_position, variants)
    scores: dict[str, np.ndarray] = {}
    counts: dict[str, dict[str, int]] = {}
    for panel in PANELS:
        body = _body_from_panel(args.benchmark / f"panel_{panel}.npz")
        scores[panel], counts[panel] = _promote_frozen_actions(
            body, policy, cache, official, row_position, variants, args.batch_queries,
        )
    method = {
        "name": "chemaware_v2_reranker",
        "kind": "frozen_candidate_action_policy",
        "policy_sha256": policy_hash,
        "policy_schema": policy["schema"],
        "truth_fields_used": [],
        "selection_counts": counts,
        "score_semantics": "official pair scores with selected candidate promoted above baseline maximum",
    }
    report = write_pair_score_cache(args.output, args.benchmark, method, scores)
    summary = {
        "status": "CHEMAWARE_V2_GNPS_SCORE_EXPORT_COMPLETE",
        "labels_opened": False,
        "benchmark": benchmark_fingerprint(args.benchmark),
        "policy_sha256": policy_hash,
        "selection_counts": counts,
        "cache_report_sha256": sha256_file(args.output / "report.json"),
    }
    (args.output / "selection_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps({"cache": report, "summary": summary}, indent=2), flush=True)


def main() -> None:
    args = arguments()
    if args.batch_queries < 1:
        raise ValueError("--batch-queries must be positive")
    policy, replay, policy_hash = _load_policy(args)
    if args.evidence is not None:
        _massspecgym(args, policy, replay, policy_hash)
    else:
        _gnps(args, policy, replay, policy_hash)


if __name__ == "__main__":
    main()
