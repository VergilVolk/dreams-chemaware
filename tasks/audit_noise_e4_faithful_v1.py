"""Fail-closed audit for the byte-exact historical E4 replay."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np


FROZEN_FILES = {
    "trainer": {
        "sha256": "28c3b375d270fc2030783938d9390710c2c5ab8f0926a0b4ff507d62afa26885",
        "git_blob_sha1": "89307d572c02f4409bcd669e088659717a72f85c",
        "manifest": "tasks/train_noise_e4_faithful_v1.py",
    },
    "validator": {
        "sha256": "95b0e6a5bcdd0a473dacc63b8b15e8f63d88b0a90ae6612acae77931330973cc",
        "git_blob_sha1": "75a2413c60faf226f22af51ad35eb0512d1f3db7",
        "manifest": "tasks/validate_noise_e4_faithful_v1.py",
    },
    "historical_test": {
        "sha256": "c20e5a9e9ec2782fa9e94f62d2aa8481c844f0af5d3a2e91975466f1f7a99ab3",
        "git_blob_sha1": "426cb242822a12f8230f42fb6a1b4dbf2a78761e",
        "manifest": "tasks/test_noise_e4_faithful_v1.py",
    },
    "noise_v3_core": {
        "sha256": "a7dd0b07ed2093f0e38b6e36d2645879853cbed9b3d1e0a51c1c7129d43b7997",
        "git_blob_sha1": "468c4482658dea6c3c52c0c24b7905ef0deb0e76",
        "manifest": "tasks/noise_e4_faithful_v1_noise_v3_core.py",
    },
    "dreams_layers": {
        "sha256": "ab5e44974e7dac2670d687812e2537e8a41c60248512c1e15e72727987536279",
        "git_blob_sha1": "05a7ee4f4778946fd10f4f44006ebe17d4390c61",
        "manifest": "dreams/models/dreams/layers_e4_historical.py",
    },
}
FROZEN_DEPENDENCIES = {
    "tasks/noise_final_core.py": "3a9c9b9e2462527c481986e2686616a95a8c02dc59d69e03309e44e0f5fc562d",
    "tasks/train_e1_identity.py": "771d035fe29a604b910f2d87ca3b7b22b4ac89a63fdf38e098a7c4467bdc3071",
    "tasks/train_noise_final_r2_shared_encoder.py": "17fe417f414d04741ce4a2c0d0933835768803576e53586828d128264c060a73",
}
FROZEN_ARTIFACTS = {
    "graph": "5f2340751c7521c5a93114e2b134d5796f157148736ad9162d545b84c11d9f71",
    "r0_report": "4f68bfd950d44d02664f67ae9d7ac6700fd3ea33ae0f914f6e623146c57749b6",
    "r0_actions": "35d52f13e4441141d622c3d60208e3ae7f11dc554bfc129fef2baa4a1cd27843",
    "historical_cache": "86b7e60194b059b839d266e9e6b52a84b185e5b7160aa934531d4c1c026a5486",
    "data": "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f",
    "official_checkpoint": "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245",
    "architecture_checkpoint": "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2",
    "corrected_graph": "8a57bb3a9cccdf69a738f2b093bfad8f6fc4393b23f1342c7a035ac54ce58fa1",
    "corrected_cache": "18d7632adc67dcda5d650a5c0bc344d8db3878f9e1ff3f8d51a6d4918b96bbda",
    "corrected_metadata": "504ce0a570ac1bc461e76cad81acb3ca3560e58b008ec2d8ff14036e944599f5",
}
FROZEN_CURRICULUM = (
    ("candidate_gradient", 0.50, 3),
    ("candidate_gradient", 0.50, 4),
    ("candidate_gradient", 0.50, 5),
    ("candidate_gradient", 0.50, 6),
    ("role_confounder", 1.00, 1),
    ("role_confounder", 1.00, 2),
    ("role_confounder", 1.00, 3),
    ("role_confounder", 1.00, 4),
    ("role_confounder", 1.00, 5),
)
FROZEN_CELL_ROWS = {
    ("candidate_gradient", 0.50, 3): 9974,
    ("candidate_gradient", 0.50, 4): 8717,
    ("candidate_gradient", 0.50, 5): 7805,
    ("candidate_gradient", 0.50, 6): 6998,
    ("role_confounder", 1.00, 1): 1092,
    ("role_confounder", 1.00, 2): 766,
    ("role_confounder", 1.00, 3): 630,
    ("role_confounder", 1.00, 4): 521,
    ("role_confounder", 1.00, 5): 431,
}
FROZEN_ARGUMENTS = {
    "policy": "curriculum",
    "action_scope": "all",
    "outer_fold": "0",
    "formula_fold_seed": "20260825",
    "epochs": "4",
    "batch_actions": "4",
    "views_per_identity": "4",
    "positive_spectra": "4",
    "negative_molecules": "8",
    "unfreeze_blocks": "1",
    "backbone_lr": "2e-6",
    "head_lr": "1e-5",
    "weight_decay": "1e-4",
    "rank_margin": "0.05",
    "temperature": "0.10",
    "lambda_clean_rank": "1.0",
    "lambda_aug_rank": "1.0",
    "lambda_consistency": "0.25",
    "lambda_margin_floor": "2.0",
    "lambda_preserve": "5.0",
    "margin_floor_slack": "0.005",
    "safety_ratio": "1.0",
    "grad_clip": "1.0",
    "run_suffix": "highlr_multifold",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    body = path.read_bytes()
    return hashlib.sha1(
        b"blob " + str(len(body)).encode() + b"\0" + body,
    ).hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--historical-test", type=Path, required=True)
    parser.add_argument("--noise-v3-core", type=Path, required=True)
    parser.add_argument("--dreams-layers", type=Path, required=True)
    parser.add_argument("--sbatch", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--r0-dir", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--historical-cache", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--official-checkpoint", type=Path)
    parser.add_argument("--architecture-checkpoint", type=Path)
    parser.add_argument("--corrected-graph", type=Path)
    parser.add_argument("--corrected-cache", type=Path)
    parser.add_argument("--corrected-metadata", type=Path)
    parser.add_argument("--runtime-report", type=Path)
    return parser.parse_args()


def literal_assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"missing literal assignment {name}")


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split(maxsplit=1)
            entries[relative.strip()] = digest
    return entries


def audit_static(args: argparse.Namespace) -> dict[str, object]:
    observed_paths = {
        "trainer": args.trainer,
        "validator": args.validator,
        "historical_test": args.historical_test,
        "noise_v3_core": args.noise_v3_core,
        "dreams_layers": args.dreams_layers,
    }
    for role, path in observed_paths.items():
        expected = FROZEN_FILES[role]
        if sha256(path) != expected["sha256"]:
            raise RuntimeError(f"{role} bytes differ from historical E4")
        if git_blob_sha1(path) != expected["git_blob_sha1"]:
            raise RuntimeError(f"{role} Git-blob identity differs from historical E4")

    tree = ast.parse(args.trainer.read_text(encoding="utf-8"))
    policy = literal_assignment(tree, "FIXED_POLICY")
    observed_curriculum = tuple(tuple(value) for value in policy["curriculum"])
    if observed_curriculum != FROZEN_CURRICULUM:
        raise RuntimeError("historical E4 curriculum was modified")

    sbatch = args.sbatch.read_text(encoding="utf-8")
    if not re.search(r"^#SBATCH\s+--gpus=2\s*$", sbatch, flags=re.MULTILINE):
        raise RuntimeError("faithful E4 job must request exactly two GPUs")
    if re.search(r"^#SBATCH\s+--mem(?:=|\s)", sbatch, flags=re.MULTILINE):
        raise RuntimeError("manual Slurm memory specification is forbidden")
    required_literals = (
        'HISTORICAL_GRAPH="data/validation/g8r_error_atlas_listwise_cache.npz"',
        'HISTORICAL_R0="data/validation/g8r_noise_final_r0_faithful_s3a"',
        'HISTORICAL_CACHE="data/validation/g8r_p2_official_embeddings.npz"',
        'cp "$SNAPSHOT_ROOT/tasks/train_noise_e4_faithful_v1.py"',
        '"$SNAPSHOT_ROOT/tasks/train_noise_final_e4a_direct_augmentation.py"',
        'cp "$SNAPSHOT_ROOT/tasks/noise_e4_faithful_v1_noise_v3_core.py"',
        '"$SNAPSHOT_ROOT/tasks/noise_v3_core.py"',
        'cp "$SNAPSHOT_ROOT/dreams/models/dreams/layers_e4_historical.py"',
        '"$SNAPSHOT_ROOT/dreams/models/dreams/layers.py"',
        'launch_replay "$GPU_ZERO" primary 20260830',
        'launch_replay "$GPU_ONE" replica 20260829',
    )
    absent = [value for value in required_literals if value not in sbatch]
    if absent:
        raise RuntimeError(f"SBATCH lacks historical E4 literals: {absent}")
    forbidden = (
        "--causal-arm", "--materialized-action-dir",
        "--materialized-injection-mode", "--optimizer-boundary-mode",
        "--injector-target-attributable-fraction", "--initial-student-checkpoint",
        "--outcome-action-dir", "--candidate-boundary-loss",
        "--positive-stream-weight", "--guided-noise-policy",
        "--direct-transfer-mode", "--rank-reference-mode",
        "matched_random", "clean_duplicate", "multi_action_balanced",
    )
    present = [value for value in forbidden if value in sbatch]
    if present:
        raise RuntimeError(f"post-E4 mechanism leaked into SBATCH: {present}")
    for name, value in FROZEN_ARGUMENTS.items():
        token = f"--{name.replace('_', '-')} {value}"
        if token not in sbatch:
            raise RuntimeError(f"SBATCH does not freeze historical E4 option: {token}")
    if "--no-amp" not in sbatch:
        raise RuntimeError("historical E4 full-fp32 contract is missing")

    entries = read_manifest(args.source_manifest)
    for role, expected in FROZEN_FILES.items():
        if entries.get(expected["manifest"]) != expected["sha256"]:
            raise RuntimeError(f"source manifest does not lock {role}")
    for path, expected in FROZEN_DEPENDENCIES.items():
        if entries.get(path) != expected:
            raise RuntimeError(f"historical dependency drifted: {path}")
    for forbidden_name in (
        "tasks/noise_action_injector_v1.py",
        "tasks/noise_e4_action_injector_v1_bridge.py",
        "tasks/noise_final_e4_pmt_core.py",
        "tasks/noise_corrected_action_routing_v3.py",
    ):
        if forbidden_name in entries:
            raise RuntimeError(f"non-E4 implementation entered source closure: {forbidden_name}")
    return {
        "historical_files": FROZEN_FILES,
        "historical_dependencies": FROZEN_DEPENDENCIES,
        "curriculum": [list(value) for value in FROZEN_CURRICULUM],
        "two_gpus": True,
        "manual_memory_request": False,
        "replays": {"primary": 20260830, "replica": 20260829},
        "post_e4_injector_or_optimizer_boundary": False,
        "source_and_registered_artifacts_byte_exact": True,
        "runtime_cuda_stack_byte_exact": False,
    }


def audit_runtime(args: argparse.Namespace) -> dict[str, object]:
    runtime_paths = (
        args.r0_dir, args.graph, args.historical_cache, args.data,
        args.official_checkpoint, args.architecture_checkpoint,
        args.corrected_graph, args.corrected_cache, args.corrected_metadata,
    )
    if all(path is None for path in runtime_paths):
        return {"performed": False}
    if any(path is None for path in runtime_paths):
        raise RuntimeError("runtime E4 audit requires the complete frozen artifact closure")
    report_path = args.r0_dir / "report.json"
    action_path = args.r0_dir / "training_actions.csv.gz"
    actual_artifacts = {
        "graph": args.graph,
        "r0_report": report_path,
        "r0_actions": action_path,
        "historical_cache": args.historical_cache,
        "data": args.data,
        "official_checkpoint": args.official_checkpoint,
        "architecture_checkpoint": args.architecture_checkpoint,
        "corrected_graph": args.corrected_graph,
        "corrected_cache": args.corrected_cache,
        "corrected_metadata": args.corrected_metadata,
    }
    for role, path in actual_artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != FROZEN_ARTIFACTS[role]:
            raise RuntimeError(
                f"historical artifact drifted: {role} {observed} != {FROZEN_ARTIFACTS[role]}"
            )

    import pandas as pd
    from train_noise_final_e4a_direct_augmentation import (
        CandidateGraph,
        parse_path,
        stable_fold,
    )
    from train_noise_final_r2_shared_encoder import parse_controls

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("formal") is not True
        or report.get("contracts", {}).get("P2b") != "forbidden"
        or report.get("contracts", {}).get(
            "action_outcomes_absent_from_training_manifest"
        ) is not True
    ):
        raise RuntimeError("R0 is not the formal outcome-free E4 action artifact")
    actions = pd.read_csv(action_path, low_memory=False)
    forbidden = {"corrected", "introduced", "target_rank", "target_margin", "random_margin"}
    if forbidden & set(actions.columns):
        raise RuntimeError("post-outcome data entered the historical action manifest")
    required = {
        "query_index", "query_row", "query_ik14", "query_formula", "formula_fold",
        "selector", "attenuation", "step", "target_path", "hard_negative_row",
        "matched_control_paths",
    }
    if required - set(actions.columns):
        raise RuntimeError(f"R0 columns are incomplete: {sorted(required - set(actions.columns))}")
    observed_cells = {
        (str(selector), float(attenuation), int(step)): int(len(block))
        for (selector, attenuation, step), block in actions.groupby(
            ["selector", "attenuation", "step"], sort=False,
        )
    }
    if observed_cells != FROZEN_CELL_ROWS or len(actions) != 36934:
        raise RuntimeError(f"R0 action population drifted: {observed_cells}")
    train = actions.loc[actions["formula_fold"].astype(int).ne(0)].copy()
    if (
        len(train) != 28509
        or train["query_ik14"].astype(str).nunique() != 1562
        or train["query_formula"].astype(str).nunique() != 689
    ):
        raise RuntimeError("historical fold-0 action population drifted")
    if not np.array_equal(
        actions["formula_fold"].to_numpy(np.int8),
        actions["query_formula"].astype(str).map(
            lambda value: stable_fold(value, 5, 20260825)
        ).to_numpy(np.int8),
    ):
        raise RuntimeError("R0 formula-fold assignments drifted")

    graph = CandidateGraph(args.graph)
    query = actions["query_index"].to_numpy(np.int64)
    if np.any((query < 0) | (query >= graph.n_queries)):
        raise RuntimeError("R0 query index is outside the historical graph")
    if not np.array_equal(actions["query_row"].to_numpy(np.int64), graph.query_row[query]):
        raise RuntimeError("R0 query rows do not reproduce historical graph")
    if not np.array_equal(actions["query_ik14"].astype(str).to_numpy(), graph.query_ik14[query]):
        raise RuntimeError("R0 identities do not reproduce historical graph")
    if not np.array_equal(actions["query_formula"].astype(str).to_numpy(), graph.query_formula[query]):
        raise RuntimeError("R0 formulas do not reproduce historical graph")
    for row in actions.itertuples(index=False):
        path = parse_path(row.target_path)
        if len(path) < int(row.step):
            raise RuntimeError("R0 target path is shorter than frozen step")
        controls = parse_controls(row.matched_control_paths)
        if len(controls) != 2:
            raise RuntimeError("R0 matched-control multiplicity is not two")
        _, candidate_rows, molecule_ptr, _ = graph.query_block(int(row.query_index))
        negative = set(map(int, candidate_rows[int(molecule_ptr[1]):]))
        if int(row.hard_negative_row) not in negative:
            raise RuntimeError("R0 hard negative does not reproduce historical geometry")
    return {
        "performed": True,
        "artifact_sha256": FROZEN_ARTIFACTS,
        "r0_rows": int(len(actions)),
        "fold0_training_rows": int(len(train)),
        "fold0_training_identities": int(train["query_ik14"].nunique()),
        "fold0_training_formulas": int(train["query_formula"].nunique()),
        "all_query_rows_identities_formulas_replay": True,
        "all_hard_negatives_replay": True,
        "all_target_paths_and_two_controls_present": True,
        "post_outcome_fields_absent": True,
    }


def main() -> None:
    args = arguments()
    result = {
        "status": "noise_e4_faithful_v1_audit_pass",
        "static": audit_static(args),
        "runtime": audit_runtime(args),
        "claim": (
            "Recovered historical E4 sources and registered artifacts are byte-locked; "
            "the CUDA/runtime stack is not, and corrected metrics are development-only."
        ),
    }
    if args.runtime_report is not None:
        args.runtime_report.parent.mkdir(parents=True, exist_ok=True)
        if args.runtime_report.exists():
            raise RuntimeError(f"refusing to overwrite {args.runtime_report}")
        args.runtime_report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
