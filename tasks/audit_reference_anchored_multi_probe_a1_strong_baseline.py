"""Post-hoc strong-baseline audit of the completed A1 coordinate screen.

The frozen A1 primary baseline included precursor mass, but that mass term was
empirically weaker than the already preregistered direct multichannel score in
both panels.  This audit does not refit or rescore anything.  It recomputes
paired IK14-cluster intervals from the frozen per-query table and decides only
whether one small official-DreaMS confirmation is worth running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    metric: str,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    import numpy as np

    delta = frame.assign(delta=(
        frame[f"{candidate}__{metric}"].astype(float)
        - frame[f"{baseline}__{metric}"].astype(float)
    )).groupby("query_ik14", sort=True).delta.mean().to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=float)
    for position in range(repeats):
        draw = rng.integers(0, len(delta), size=len(delta))
        samples[position] = float(delta[draw].mean())
    return {
        "mean_delta": float(delta.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "identity_clusters": int(len(delta)),
        "resamples": int(repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a1-dir", type=Path,
        default=ROOT / "data/validation/reference_anchored_multi_probe_a1_20260913",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/reference_anchored_multi_probe_a1_strong_baseline_20260913.json",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite existing audit: {args.output}")
    source_report_path = args.a1_dir / "report.json"
    per_query_path = args.a1_dir / "per_query_coordinate_recovery.csv.gz"
    for path in (source_report_path, per_query_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source.get("status") != "reference_anchored_multi_probe_a1_complete":
        raise RuntimeError("unexpected frozen A1 status")
    if source.get("formal") is not False:
        raise RuntimeError("A1 source must remain exploratory")
    if not source["gates"].get("pass_to_official_dreams_confirmation"):
        raise RuntimeError("frozen A1 did not pass its preregistered gate")
    if sha256(per_query_path) != source.get("provenance", {}).get("per_query"):
        raise RuntimeError("frozen A1 per-query hash differs from report")
    a1_script = ROOT / "tasks/pilot_reference_anchored_multi_probe_a1.py"
    if sha256(a1_script) != source.get("provenance", {}).get("script"):
        raise RuntimeError("current A1 implementation differs from the frozen source run")
    frame = pd.read_csv(per_query_path)
    if len(frame) != int(source.get("identities", -1)):
        raise RuntimeError("frozen A1 identity count mismatch")
    if frame[["panel", "query_ik14"]].duplicated().any():
        raise RuntimeError("frozen A1 contains duplicate panel/query identities")
    candidates = ("multi_anchor_profile", "profile_fusion_50_50", "multi_anchor_augmented")
    metrics = ("selected_structural_similarity", "top3_structural_hit", "ndcg5", "candidate_spearman")
    comparisons = {}
    for candidate in candidates:
        for metric in metrics:
            key = f"{candidate}_vs_direct_multichannel__{metric}"
            comparisons[key] = bootstrap(
                frame, candidate, "direct_multichannel", metric,
                args.bootstrap_resamples, args.seed + len(comparisons),
            )
    # A1 is the development screen and A1b is the independent embedding
    # confirmation. The fixed 50:50 direct+profile candidate is the mechanism-
    # aligned A1 winner; selecting it here is allowed only because A1b has not
    # been observed and its weight will not be tuned there.
    primary_candidate = "profile_fusion_50_50"
    primary = comparisons[f"{primary_candidate}_vs_direct_multichannel__ndcg5"]
    panel_deltas = {}
    for panel, group in frame.groupby("panel", sort=True):
        panel_deltas[str(panel)] = {
            candidate: float(
                group[f"{candidate}__ndcg5"].mean()
                - group["direct_multichannel__ndcg5"].mean()
            )
            for candidate in candidates
        }
    every_panel = all(values[primary_candidate] >= 0 for values in panel_deltas.values())
    report = {
        "status": "reference_anchored_multi_probe_a1_strong_baseline_audit_complete",
        "formal": False,
        "source_gate_passed": True,
        "reason": (
            "The original direct+mass baseline was weaker than direct_multichannel in both panels; "
            "this frozen-score audit compares the fixed A1 direct+profile candidate against "
            "direct_multichannel without refitting."
        ),
        "primary_candidate_frozen_for_a1b": primary_candidate,
        "comparisons": comparisons,
        "panel_ndcg_deltas_vs_direct_multichannel": panel_deltas,
        "gates": {
            "primary_ndcg_identity_ci_low_positive": bool(primary["ci_low"] > 0),
            "primary_ndcg_nonnegative_in_each_panel": bool(every_panel),
            "worth_one_official_dreams_confirmation": bool(primary["ci_low"] > 0 and every_panel),
        },
        "claim_limit": (
            "Post-hoc baseline-strength audit of frozen A1 scores. It cannot upgrade A1 to a formal result; "
            "it only controls whether the bounded A1b confirmation is run."
        ),
        "provenance": {
            "source_report_sha256": sha256(source_report_path),
            "source_per_query_sha256": sha256(per_query_path),
            "source_a1_script_sha256": sha256(a1_script),
            "audit_script_sha256": sha256(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
