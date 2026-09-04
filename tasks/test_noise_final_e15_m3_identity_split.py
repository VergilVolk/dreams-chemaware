"""Fast selection tests for the E15-M3 identity-held split."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import build_noise_final_e15_m3_identity_split as split


class Graph:
    n_queries = 600
    query_has_near = np.asarray([(index % 2) == 0 for index in range(n_queries)])


def frame(kind: str) -> pd.DataFrame:
    rows = []
    for source_index, source in enumerate(split.SOURCES):
        for local in range(80):
            query = 100 * source_index + local
            rows.append({
                "source": source, "query_index": query,
                "query_ik14": f"{kind}-{source}-{local}",
                "query_formula": f"F{local % 20}", "action_id": f"a-{query}",
                "source_kind_percentile": (local + 1) / 80,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ranks = np.full(Graph.n_queries, 2, dtype=np.int16)
    errors = split.held_errors(frame("e"), ranks, Graph(), 32, 1)
    if len(errors) != 128 or errors["query_ik14"].nunique() != 128:
        raise RuntimeError("held-error selection is not identity unique")
    ranks[:] = 1
    correct = split.held_correct(
        frame("c"), ranks, Graph(), set(errors["query_ik14"].astype(str)), 128, 1,
    )
    if len(correct) != 128 or correct["query_ik14"].nunique() != 128:
        raise RuntimeError("held-correct selection is not identity unique")
    error_counts = errors.groupby("source").size().to_dict()
    if set(error_counts.values()) != {32} or set(error_counts) != set(split.SOURCES):
        raise RuntimeError(f"held errors are not source balanced: {error_counts}")
    sbatch = (Path(__file__).with_name("run_noise_final_e15_m3_identity_split.sbatch")).read_text(
        encoding="utf-8",
    )
    required = (
        "#SBATCH --gpus=1", "set -euo pipefail", "run_${SLURM_JOB_ID}",
        "build_noise_final_e15_m3_identity_split.py",
        "validate_noise_final_e15_m3_identity_split.py",
    )
    if any(token not in sbatch for token in required) or "#SBATCH --mem" in sbatch:
        raise RuntimeError("M3 sbatch violates the single-GPU fail-closed contract")
    builder = Path(split.__file__).read_text(encoding="utf-8")
    if "initial_student_decision" in builder or "pass_to_multifold" in builder:
        raise RuntimeError("M3 builder still depends on version-sensitive legacy decision fields")
    for key in ("initial_student_checkpoint", "official_checkpoint", "architecture_checkpoint"):
        if f'"{key}": sha256_file' not in builder:
            raise RuntimeError(f"M3 builder does not bind {key} to passing M2 provenance")
    forbidden_arbitrary_gates = (
        "held_correct_sources_balanced", "train_corrective_queries_ge_1000",
        "train_corrective_identities_ge_500", "train_harmful_queries_ge_500",
        "held_correct_sources_represented", "all_harmful_sources_remain_nonempty",
    )
    if any(token in builder for token in forbidden_arbitrary_gates):
        raise RuntimeError("M3 builder contains an uncalibrated absolute capacity gate")
    print("[test_noise_final_e15_m3_identity_split] PASS", flush=True)


if __name__ == "__main__":
    main()
