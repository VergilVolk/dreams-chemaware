#!/usr/bin/env python
"""Materialize candidate-molecule score archives for the GNPS run-16 ladder.

This is a consumed-development smoke input for the generic mega evaluator.  It
does not create a new external result.  The fifteen historical methods are
max-pooled from their frozen query-reference-spectrum scores with the panel's
``molecule_ptr``.  Full23 v1.3.2 is reconstructed exactly as

    Full23_v1 + 0.9 * (Full23_v1.3.1 - Full23_v1).

The output score NPZ is bound to the exact panel SHA and records N_Q/N_E/N_P.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAIR_SCORES = ROOT / "data/validation/GLM_gnps_article_benchmark_s2v26/run15/bundle/method_scores.npz"
V1_PREDICTIONS = ROOT / "data/validation/full23_tensor_set_readout_run_2433126/gnps_development_evaluation"
V131_PREDICTIONS = ROOT / "data/validation/full23_v131_direct_repairs_run_2434102/gnps_development_evaluation"
PANELS = {
    "identity_disjoint": ROOT / "data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz",
    "formula_disjoint": ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1/panel_formula_disjoint.npz",
}
RESIDUAL_SCALE = 0.9


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/gnps_run16_mega_evaluation_inputs_20261102",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def pool_reference_scores(pair_scores: np.ndarray, molecule_ptr: np.ndarray) -> np.ndarray:
    if pair_scores.ndim != 2:
        raise RuntimeError(f"pair score matrix must be two-dimensional: {pair_scores.shape}")
    if molecule_ptr.ndim != 1 or molecule_ptr.dtype.kind not in "iu":
        raise RuntimeError("molecule_ptr must be a one-dimensional integer array")
    if int(molecule_ptr[0]) != 0 or int(molecule_ptr[-1]) != pair_scores.shape[1]:
        raise RuntimeError("molecule_ptr does not delimit the frozen pair score axis")
    if np.any(np.diff(molecule_ptr) < 1):
        raise RuntimeError("every molecule edge must have at least one reference spectrum")
    starts = np.asarray(molecule_ptr[:-1], dtype=np.int64)
    return np.maximum.reduceat(np.asarray(pair_scores, dtype=np.float32), starts, axis=1)


def build_panel_scores(
    panel_name: str,
    panel_path: Path,
    pair_archive: np.lib.npyio.NpzFile,
    output: Path,
) -> dict[str, object]:
    with np.load(panel_path, allow_pickle=False) as panel:
        required = {
            "query_ptr", "molecule_ptr", "molecule_label", "molecule_ik14",
            "candidate_row", "query_row", "query_ik14", "query_formula",
        }
        missing = sorted(required.difference(panel.files))
        if missing:
            raise RuntimeError(f"{panel_name} panel is incomplete: {missing}")
        query_ptr = np.asarray(panel["query_ptr"], dtype=np.int64)
        molecule_ptr = np.asarray(panel["molecule_ptr"], dtype=np.int64)
        labels = np.asarray(panel["molecule_label"], dtype=np.int8)
        molecule_ik14 = np.asarray(panel["molecule_ik14"])
        n_q = len(query_ptr) - 1
        n_e = len(labels)
        n_p = int(molecule_ptr[-1])

        pair_key = f"scores_{panel_name}"
        pair_matrix = np.asarray(pair_archive[pair_key], dtype=np.float32)
        historical_names = np.asarray(pair_archive["method_names"])
        pooled = pool_reference_scores(pair_matrix, molecule_ptr)
        if pooled.shape != (len(historical_names), n_e):
            raise RuntimeError(f"unexpected pooled shape: {pooled.shape}")

        v1_path = V1_PREDICTIONS / f"candidate_predictions_{panel_name}.npz"
        v131_path = V131_PREDICTIONS / f"candidate_predictions_{panel_name}.npz"
        require_file(v1_path)
        require_file(v131_path)
        with np.load(v1_path, allow_pickle=False) as v1, np.load(v131_path, allow_pickle=False) as v131:
            for archive_name, archive in (("Full23 v1", v1), ("Full23 v1.3.1", v131)):
                if not np.array_equal(archive["query_ptr"], query_ptr):
                    raise RuntimeError(f"{archive_name} query axis drifted on {panel_name}")
                if not np.array_equal(np.asarray(archive["labels"], dtype=np.int8), labels):
                    raise RuntimeError(f"{archive_name} labels drifted on {panel_name}")
            if "candidate_ids" in v1.files and not np.array_equal(v1["candidate_ids"], molecule_ik14):
                raise RuntimeError(f"Full23 v1 candidate identities drifted on {panel_name}")
            full23_v1 = np.asarray(v1["full23_v1"], dtype=np.float32)
            full23_v131 = np.asarray(v131["full23_v131"], dtype=np.float32)
        if full23_v1.shape != (n_e,) or full23_v131.shape != (n_e,):
            raise RuntimeError("Full23 candidate score vector has the wrong length")
        full23_v132 = full23_v1 + np.float32(RESIDUAL_SCALE) * (full23_v131 - full23_v1)

    names = np.concatenate([historical_names.astype(str), np.asarray(["full23_v132"])])
    molecule_scores = np.concatenate([pooled, full23_v132[None, :]], axis=0).astype(np.float32)
    score_path = output / f"scores_{panel_name}.npz"
    np.savez_compressed(
        score_path,
        schema_version=np.asarray("mega_identity_score_archive_v1"),
        method_names=names,
        molecule_scores=molecule_scores,
        panel_sha256=np.asarray(sha256(panel_path)),
        N_Q=np.asarray(n_q, dtype=np.int64),
        N_E=np.asarray(n_e, dtype=np.int64),
        N_P=np.asarray(n_p, dtype=np.int64),
    )
    return {
        "panel": panel_name,
        "panel_path": str(panel_path),
        "panel_sha256": sha256(panel_path),
        "score_path": str(score_path),
        "score_sha256": sha256(score_path),
        "N_Q": n_q,
        "N_E": n_e,
        "N_P": n_p,
        "methods": names.tolist(),
        "full23_v132_formula": "full23_v1 + 0.9 * (full23_v131 - full23_v1)",
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = arguments(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for path in [PAIR_SCORES, *PANELS.values()]:
        require_file(path)
    output.mkdir(parents=True)
    try:
        with np.load(PAIR_SCORES, allow_pickle=False) as pair_archive:
            reports = [
                build_panel_scores(name, path, pair_archive, output)
                for name, path in PANELS.items()
            ]
        report = {
            "status": "GNPS_RUN16_MEGA_EVALUATION_INPUTS_COMPLETE",
            "dataset_role": "consumed development smoke and same-axis comparison",
            "allow_external_claim": False,
            "pair_score_source": str(PAIR_SCORES),
            "pair_score_sha256": sha256(PAIR_SCORES),
            "residual_scale": RESIDUAL_SCALE,
            "panels": reports,
        }
        with (output / "report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except Exception:
        for path in output.iterdir():
            if path.is_file():
                path.unlink()
        output.rmdir()
        raise


if __name__ == "__main__":
    main()
