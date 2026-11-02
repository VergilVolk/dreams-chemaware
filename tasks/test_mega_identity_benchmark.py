#!/usr/bin/env python
"""Synthetic formal-contract test for the mega identity evaluator/plotter."""
from __future__ import annotations

import gzip
import json
import math
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

from evaluate_mega_identity_benchmark import (
    NpzMemmapStore,
    evaluate,
    load_and_validate_panel,
    sha256_file,
)
from plot_mega_identity_benchmark import plot, stream_curve


def assert_close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{actual!r} != {expected!r}")


def panel_axis_digest(panel_path: Path, work: Path) -> str:
    extraction = work / "axis_extract"
    extraction.mkdir()
    store = NpzMemmapStore(panel_path, extraction, "panel")
    try:
        return load_and_validate_panel(store).axis_digest
    finally:
        store.close()
        shutil.rmtree(extraction, ignore_errors=True)


def test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        panel_path = root / "panel.npz"
        labels = np.asarray([1, 0, 0, 0, 1, 0, 0, 1], dtype=bool)
        query_ik14 = np.asarray(
            ["AAAAAAAAAAAAAA", "DDDDDDDDDDDDDD", "GGGGGGGGGGGGGG"], dtype="U14"
        )
        molecule_ik14 = np.asarray(
            [
                "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC",
                "EEEEEEEEEEEEEE", "DDDDDDDDDDDDDD", "FFFFFFFFFFFFFF",
                "HHHHHHHHHHHHHH", "GGGGGGGGGGGGGG",
            ],
            dtype="U14",
        )
        query_formula = np.asarray(["C2H6O", "C2H6O", "C3H8"], dtype="U16")
        molecule_formula = np.asarray(
            ["C2H6O", "C2H6O", "C2H4O", "C2H6O", "C2H6O", "CH4O", "C2H6", "C3H8"],
            dtype="U16",
        )
        molecule_ptr = np.asarray([0, 2, 3, 5, 6, 8, 9, 11, 13], dtype=np.int64)
        candidate_row = np.arange(100, 113, dtype=np.int64)
        panel_payload = {
            "query_ptr": np.asarray([0, 3, 6, 8], dtype=np.int64),
            "molecule_ptr": molecule_ptr,
            "candidate_row": candidate_row,
            "molecule_label": labels,
            "query_ik14": query_ik14,
            "molecule_ik14": molecule_ik14,
            "query_formula": query_formula,
            "molecule_formula": molecule_formula,
            "near_query": np.asarray([1, 1, 0], dtype=bool),
            "query_row": np.asarray([10, 20, 30], dtype=np.int64),
        }
        np.savez_compressed(panel_path, **panel_payload)

        bad_ptr_panel = root / "bad_unsigned_ptr_panel.npz"
        np.savez_compressed(
            bad_ptr_panel,
            **{
                **panel_payload,
                "molecule_ptr": np.asarray([0, 2, 1, 5, 6, 8, 9, 11, 13], dtype=np.uint64),
            },
        )
        try:
            panel_axis_digest(bad_ptr_panel, root)
        except RuntimeError as error:
            assert "candidate molecule" in str(error) or "molecule_ptr" in str(error)
        else:
            raise AssertionError("descending unsigned molecule_ptr was accepted")

        whitespace_panel = root / "whitespace_panel.npz"
        whitespace_ids = query_ik14.astype("U15")
        whitespace_ids[0] = "AAAAAAAAAAAAAA "
        np.savez_compressed(
            whitespace_panel, **{**panel_payload, "query_ik14": whitespace_ids}
        )
        try:
            panel_axis_digest(whitespace_panel, root)
        except RuntimeError as error:
            assert "whitespace" in str(error)
        else:
            raise AssertionError("whitespace-distinct identity cluster was accepted")
        axis_digest = panel_axis_digest(panel_path, root)
        panel_hash = sha256_file(panel_path)
        scores_path = root / "scores.npz"
        np.savez_compressed(
            scores_path,
            panel_sha256=np.asarray(panel_hash),
            axis_digest=np.asarray(axis_digest),
            N_Q=np.asarray(3, dtype=np.int64),
            N_E=np.asarray(8, dtype=np.int64),
            N_P=np.asarray(13, dtype=np.int64),
            method_names=np.asarray(["alpha", "beta"], dtype="U16"),
            molecule_scores=np.asarray(
                [
                    [0.9, 0.2, 0.1, 0.8, 0.8, 0.1, 0.7, 0.5],
                    [0.4, 0.5, 0.1, 0.2, 0.9, 0.1, 0.3, 0.8],
                ],
                dtype=np.float32,
            ),
        )

        # Same score length/counts but wrong axis provenance must be rejected.
        wrong_scores = root / "wrong_axis_scores.npz"
        np.savez_compressed(
            wrong_scores,
            panel_sha256=np.asarray("0" * 64),
            N_Q=np.asarray(3), N_E=np.asarray(8), N_P=np.asarray(13),
            method_names=np.asarray(["alpha"], dtype="U16"),
            molecule_scores=np.zeros((1, 8), dtype=np.float32),
        )
        try:
            evaluate(Namespace(
                panel=panel_path, scores=wrong_scores, output_dir=root / "wrong_eval",
                method=[], reference_method=None, bootstrap_resamples=10_000,
                bootstrap_seed=7, work_dir=root, sort_chunk_size=3,
                validation_chunk_size=3, progress_every=0,
            ))
        except RuntimeError as error:
            assert "panel_sha256" in str(error)
        else:
            raise AssertionError("same-length wrong-axis scores were accepted")

        mutated_panel = root / "mutated_query_axis_panel.npz"
        np.savez_compressed(
            mutated_panel,
            **{
                **panel_payload,
                "query_row": np.asarray([11, 20, 30], dtype=np.int64),
            },
        )
        stale_axis_scores = root / "stale_axis_scores.npz"
        np.savez_compressed(
            stale_axis_scores,
            axis_digest=np.asarray(axis_digest),
            N_Q=np.asarray(3), N_E=np.asarray(8), N_P=np.asarray(13),
            method_names=np.asarray(["alpha"], dtype="U16"),
            molecule_scores=np.zeros((1, 8), dtype=np.float32),
        )
        try:
            evaluate(Namespace(
                panel=mutated_panel, scores=stale_axis_scores,
                output_dir=root / "stale_axis_eval", method=[], reference_method=None,
                bootstrap_resamples=10_000, bootstrap_seed=7, work_dir=root,
                sort_chunk_size=3, validation_chunk_size=3, progress_every=0,
            ))
        except RuntimeError as error:
            assert "axis_digest" in str(error)
        else:
            raise AssertionError("stale query-row axis digest was accepted")

        evaluation = root / "evaluation"
        report = evaluate(Namespace(
            panel=panel_path, scores=scores_path, output_dir=evaluation,
            method=[], reference_method=None, bootstrap_resamples=10_000,
            bootstrap_seed=20261102, work_dir=root, sort_chunk_size=3,
            validation_chunk_size=3, progress_every=0,
        ))
        assert report["status"] == "complete"
        assert report["open_set_claim"] is False
        assert report["panel"]["N_Q"] == 3
        assert report["panel"]["N_E"] == 8
        assert report["panel"]["N_P"] == 13
        assert report["panel"]["N_P"] > report["panel"]["N_E"] > report["panel"]["N_Q"]
        assert report["panel"]["N_I"] == 3
        assert report["panel"]["N_F"] == 2
        assert report["scores"]["validated_contract"]["axis_digest"] == axis_digest

        alpha = report["methods"]["alpha"]
        retrieval = alpha["retrieval"]
        assert_close(retrieval["recall@1"], 1 / 3)
        assert_close(retrieval["recall@2"], 1.0)
        assert_close(retrieval["mrr"], 2 / 3)
        assert_close(retrieval["mean_rank"], 5 / 3)
        assert_close(retrieval["median_rank"], 2.0)
        assert_close(retrieval["macro_query_auroc"], 7 / 12)
        assert_close(retrieval["macro_query_average_precision"], 2 / 3)
        assert_close(retrieval["macro_query_historical_auprc"], 2 / 3)
        assert_close(alpha["micro_candidate"]["auroc"], 5 / 6)
        assert_close(alpha["micro_candidate"]["average_precision"], 34 / 45)
        assert alpha["near_subset"] is not None
        for cutoff in (1, 2, 3, 5, 10, 20):
            assert f"recall@{cutoff}" in alpha["near_subset"]
        for key in (
            "mrr", "mean_rank", "median_rank", "macro_query_auroc",
            "macro_query_average_precision", "macro_query_historical_auprc",
        ):
            assert key in alpha["near_subset"]

        beta = report["methods"]["beta"]
        assert_close(beta["retrieval"]["recall@1"], 2 / 3)
        assert_close(beta["micro_candidate"]["auroc"], 14 / 15)
        assert_close(beta["micro_candidate"]["average_precision"], 11 / 12)
        paired = report["paired_comparisons"]["beta"]
        assert paired["corrected"] == 2
        assert paired["introduced"] == 1
        assert paired["near_corrected"] == 1
        assert paired["near_introduced"] == 1
        for cluster in ("formula_cluster_bootstrap", "identity_cluster_bootstrap"):
            assert paired[cluster]["bootstrap_resamples"] == 10_000
            assert paired[cluster]["familywise_hypotheses"] == 20
            assert set(paired[cluster]["metrics"]) == {
                "recall@1", "recall@2", "recall@3", "recall@5", "recall@10",
                "recall@20", "mrr", "mean_rank_improvement", "macro_query_auroc",
                "macro_query_average_precision",
            }

        with (evaluation / "report.json").open("r", encoding="utf-8") as handle:
            disk_report = json.load(handle)
        for method in ("alpha", "beta"):
            curves = disk_report["methods"][method]["curves"]
            for key in ("roc_csv_gz", "pr_csv_gz"):
                path = evaluation / curves[key]
                assert path.is_file() and path.suffix == ".gz"
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    final = handle.read().splitlines()[-1].split(",")
                if key == "roc_csv_gz":
                    assert_close(float(final[1]), 1.0)
                    assert_close(float(final[2]), 1.0)
                else:
                    assert_close(float(final[1]), 1.0)

        # Front-preserving reader must keep both endpoints under a tight budget.
        alpha_roc = evaluation / disk_report["methods"]["alpha"]["curves"]["roc_csv_gz"]
        curve_x, curve_y = stream_curve(
            alpha_roc, "false_positive_rate", "true_positive_rate",
            disk_report["methods"]["alpha"]["micro_candidate"]["curve_points"],
            8, 0.25,
        )
        assert (curve_x[0], curve_y[0]) == (0.0, 0.0)
        assert (curve_x[-1], curve_y[-1]) == (1.0, 1.0)
        assert len(curve_x) == disk_report["methods"]["alpha"]["micro_candidate"]["curve_points"]

        spike_curve = root / "spike_pr.csv.gz"
        with gzip.open(spike_curve, "wt", encoding="utf-8", newline="") as handle:
            handle.write("recall,precision\n")
            for index in range(21):
                precision = 0.05 if index == 5 else 0.8
                handle.write(f"{index / 20:.6f},{precision:.6f}\n")
        spike_x, spike_y = stream_curve(
            spike_curve, "recall", "precision", 21, 8, 0.5
        )
        assert len(spike_x) <= 8
        assert 0.05 in spike_y  # bucket min preserves a narrow PR-front drop

        figures = root / "figures"
        plot_report = plot(Namespace(
            evaluation_dir=evaluation, output_dir=figures, max_curve_points=8,
            dpi=100, roc_front_fpr=0.25, pr_front_recall=0.34,
            title="Synthetic identity contract",
        ))
        assert plot_report["status"] == "complete"
        assert plot_report["ladder_order"] == ["beta", "alpha"]
        for name in ("ladder", "topk", "roc", "pr"):
            for extension in ("png", "pdf"):
                path = figures / f"{name}.{extension}"
                assert path.is_file() and path.stat().st_size > 0


def main() -> None:
    test()
    print("[test_mega_identity_benchmark] PASS")


if __name__ == "__main__":
    main()
