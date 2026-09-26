"""Freeze the reproducible ChemAware +2.1266 pp role-2 embedding artifact.

This is deliberately a development-result archive, not a role-3 release.  The
script refuses to label an artifact unless the frozen role-2 evaluation exactly
recovers the observed top-1 transition counts and at least +2 percentage points
over the official embedding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


EXPECTED_QUERIES = 1975
EXPECTED_CORRECTED = 54
EXPECTED_INTRODUCED = 12
EXPECTED_DELTA_RECALL1 = (EXPECTED_CORRECTED - EXPECTED_INTRODUCED) / EXPECTED_QUERIES


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--triplet-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="phaseA_step-002000")
    parser.add_argument("--official-name", default="official")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def row_by_name(report: dict[str, object], name: str) -> dict[str, object]:
    rows = [row for row in report["results"] if row["name"] == name]
    if len(rows) != 1:
        raise RuntimeError(f"evaluation must contain exactly one {name!r} row")
    return rows[0]


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite protected artifact: {args.output}")
    for path in (args.checkpoint, args.evaluation, args.triplet_report):
        if not path.is_file():
            raise FileNotFoundError(path)

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if int(evaluation.get("formula_role", -1)) != 2:
        raise RuntimeError("the +2.1266 pp artifact must be evaluated on frozen role 2")
    official = row_by_name(evaluation, args.official_name)
    model = row_by_name(evaluation, args.model_name)
    evaluated_checkpoint = Path(model["checkpoint"]).resolve()
    if evaluated_checkpoint != args.checkpoint.resolve():
        raise RuntimeError("evaluation checkpoint does not match the artifact being frozen")
    paired = model.get("paired_vs_official")
    if not isinstance(paired, dict):
        raise RuntimeError("model lacks a paired comparison against official")

    official_metrics = official["metrics"]
    metrics = model["metrics"]
    delta = float(paired["delta_recall1"])
    ci = list(map(float, paired["formula_cluster_bootstrap_delta_recall1_ci95"]))
    corrected = int(paired["corrected_at_1"])
    introduced = int(paired["introduced_at_1"])
    gates = {
        "exact_query_count": int(metrics["queries"]) == EXPECTED_QUERIES,
        "exact_transition_counts": (
            corrected == EXPECTED_CORRECTED and introduced == EXPECTED_INTRODUCED
        ),
        "exact_recall1_delta": abs(delta - EXPECTED_DELTA_RECALL1) <= 1e-12,
        "at_least_two_percentage_points": delta >= 0.02,
        "formula_cluster_ci_strictly_positive": ci[0] > 0.0,
        "mrr_improves": float(metrics["mrr"]) > float(official_metrics["mrr"]),
        "recall3_nonnegative": (
            float(metrics["recall3"]) >= float(official_metrics["recall3"])
        ),
        "micro_auc_improves": (
            float(metrics["micro_auc"]) > float(official_metrics["micro_auc"])
        ),
        "macro_auc_improves": (
            float(metrics["macro_auc"]) > float(official_metrics["macro_auc"])
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"refusing to freeze a non-matching +2.1266 pp result: {gates}")

    checkpoint_sha256 = sha256(args.checkpoint)
    evaluation_sha256 = sha256(args.evaluation)
    triplet_report_sha256 = sha256(args.triplet_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_phasea_2pp_", dir=args.output.parent))
    try:
        checkpoint_name = "chemaware_phasea_max_boundary_step2000.ckpt"
        link_mode = link_or_copy(args.checkpoint, temporary / checkpoint_name)
        shutil.copy2(args.evaluation, temporary / "role2_checkpoint_evaluation.json")
        shutil.copy2(args.triplet_report, temporary / "triplet_report.json")
        manifest = {
            "status": "CHEMAWARE_PHASEA_2PP_ROLE2_ARTIFACT_PROTECTED",
            "claim_status": "ROLE2_DEVELOPMENT_RESULT_NOT_ROLE3_CONFIRMED",
            "shared_embedding": True,
            "reranker": False,
            "distillation": False,
            "checkpoint_file": checkpoint_name,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_storage": link_mode,
            "evaluation_sha256": evaluation_sha256,
            "triplet_report_sha256": triplet_report_sha256,
            "formula_role": 2,
            "queries": EXPECTED_QUERIES,
            "official_recall1": float(official_metrics["recall1"]),
            "model_recall1": float(metrics["recall1"]),
            "delta_recall1": delta,
            "delta_recall1_percentage_points": 100.0 * delta,
            "corrected_at_1": corrected,
            "introduced_at_1": introduced,
            "formula_cluster_bootstrap_delta_recall1_ci95": ci,
            "metrics": metrics,
            "gates": gates,
            "role3_evaluated": False,
            "outer_role4_accessed": False,
            "allowed_claim": (
                "On frozen formula role 2, the ChemAware Phase-A shared embedding "
                "improved Recall@1 over official DreaMS by 2.1266 percentage points."
            ),
            "forbidden_claim": (
                "Do not describe this artifact as independently role-3 confirmed, "
                "outer-tested, or a released replacement for the +1.8144 pp model."
            ),
        }
        (temporary / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
        checksums = {
            checkpoint_name: sha256(temporary / checkpoint_name),
            "role2_checkpoint_evaluation.json": sha256(
                temporary / "role2_checkpoint_evaluation.json"
            ),
            "triplet_report.json": sha256(temporary / "triplet_report.json"),
        }
        (temporary / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
            encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
