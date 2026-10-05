"""Independent validator for current-geometry full-action replay."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    required = [args.output_dir / name for name in (
        "report.json", "current_geometry_outcomes.csv.gz", "source_summary.csv",
        "current_geometry_embeddings.npz",
    )]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    report = json.loads(required[0].read_text(encoding="utf-8"))
    frame = pd.read_csv(required[1], low_memory=False)
    with np.load(required[3]) as body:
        embedding_rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    numeric = ["clean_margin", "target_margin", "control_margin", "paired_advantage"]
    contracts = report.get("contracts", {})
    gates = {
        "status": report.get("status") == "noise_final_dynamic_direct_current_geometry_replay_complete",
        "formal": report.get("formal") is True,
        "all_actions": len(frame) == int(report.get("actions", -1)),
        "unique_actions": not frame["action_id"].duplicated().any(),
        "finite": np.isfinite(frame[numeric].to_numpy(float)).all(),
        "advantage_exact": np.allclose(
            frame["paired_advantage"], frame["target_margin"] - frame["control_margin"], atol=2e-6,
        ),
        "sources": set(frame["source"]) == {"N", "P_intensity", "P_transfer"},
        "families": frame["family"].nunique() == 8,
        "embedding_rows_unique": len(embedding_rows) == len(np.unique(embedding_rows)),
        "embeddings_finite": embeddings.ndim == 2 and np.isfinite(embeddings).all(),
        "no_update": report.get("optimizer_steps") == 0 and contracts.get("optimizer_steps") == 0,
        "full_candidates": contracts.get("full_candidate_scoring") is True,
        "shared_encoder": contracts.get("same_encoder_for_query_and_references") is True,
        "P2b": contracts.get("P2b") == "forbidden",
        "P3": contracts.get("P3_consumed") is False,
        "pass": report.get("pass_to_current_geometry_crossfit") is True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"dynamic-direct replay validation failed: {gates}")
    print(f"[validate_noise_final_dynamic_direct_replay] PASS actions={len(frame):,}")


if __name__ == "__main__":
    main()
