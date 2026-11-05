#!/usr/bin/env python3
"""CPU-only synthetic contract for the frozen CALMS correction landscape."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_calms_correction_landscape import run


METHODS = ("official_dreams", "weighted_spectral_entropy", "full23_v132")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_scores(correct: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    scores = np.empty((len(correct), 2), dtype=np.float32)
    for index, (is_correct, gap) in enumerate(zip(correct, gaps)):
        if is_correct:
            scores[index] = (0.8, 0.8 - gap)
        else:
            scores[index] = (0.8 - gap, 0.8)
    return scores.reshape(-1)


def write_panel(root: Path, panel_name: str, row_offset: int) -> tuple[Path, pd.DataFrame]:
    n_queries = 8
    query_rows = row_offset + np.arange(n_queries, dtype=np.int64)
    query_ptr = np.arange(0, 2 * n_queries + 1, 2, dtype=np.int64)
    molecule_ptr = np.arange(0, 2 * n_queries + 1, dtype=np.int64)
    query_ik14 = np.asarray([f"IK{index // 2:012d}" for index in range(n_queries)])
    query_formula = np.asarray([f"C{index // 2 + 1}H{2 * (index // 2 + 1)}" for index in range(n_queries)])
    labels = np.tile(np.asarray([True, False]), n_queries)
    panel_path = root / f"panel_{panel_name}.npz"
    np.savez_compressed(
        panel_path,
        query_row=query_rows,
        query_ik14=query_ik14,
        query_formula=query_formula,
        query_precursor_mz=200.0 + np.arange(n_queries),
        query_ptr=query_ptr,
        molecule_ik14=np.asarray(
            [value for ik14 in query_ik14 for value in (ik14, f"N{ik14[1:]}")]
        ),
        molecule_formula=np.asarray(
            [value for formula in query_formula for value in (formula, formula)]
        ),
        molecule_label=labels,
        molecule_ptr=molecule_ptr,
        candidate_row=np.arange(2 * n_queries, dtype=np.int64),
        near_query=np.asarray([False, True] * 4),
    )
    manifest = pd.DataFrame(
        {
            "row": query_rows,
            "ik14": query_ik14,
            "formula": query_formula,
            "precursor_mz": 200.0 + np.arange(n_queries),
            "library_quality": 1,
            "n_peaks": np.asarray([30, 45, 55, 80, 105, 140, 220, 300]),
            "instrument": np.asarray(
                ["LC-ESI-Orbitrap", "LC-ESI-ToF"] * 4
            ),
        }
    )
    return panel_path, manifest


def query_metrics(
    method: str,
    scores: np.ndarray,
    panel_path: Path,
) -> pd.DataFrame:
    with np.load(panel_path, allow_pickle=False) as panel:
        query_ptr = np.asarray(panel["query_ptr"], dtype=np.int64)
        query_rows = np.asarray(panel["query_row"], dtype=np.int64)
        query_ik14 = np.asarray(panel["query_ik14"]).astype(str)
        query_formula = np.asarray(panel["query_formula"]).astype(str)
        near = np.asarray(panel["near_query"], dtype=bool)
        molecule_ptr = np.asarray(panel["molecule_ptr"], dtype=np.int64)
    rows = []
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left, right = int(left), int(right)
        values = np.asarray(scores[left:right], dtype=np.float64)
        positive = float(values[0])
        negative = float(values[1])
        rank = 1 + int(negative >= positive)
        rows.append(
            {
                "method": method,
                "query_index": query,
                "query_row": query_rows[query],
                "query_ik14": query_ik14[query],
                "query_formula": query_formula[query],
                "near": near[query],
                "candidate_molecule_edges": right - left,
                "candidate_pair_edges": molecule_ptr[right] - molecule_ptr[left],
                "positive_molecule_edge": left,
                "positive_score": positive,
                "best_negative_score": negative,
                "positive_vs_best_negative_margin": positive - negative,
                "rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "macro_query_auroc": float(positive > negative)
                + 0.5 * float(positive == negative),
            }
        )
    return pd.DataFrame.from_records(rows)


def write_inputs(root: Path) -> argparse.Namespace:
    evaluation_root = root / "evaluation"
    score_root = root / "scores"
    evaluation_root.mkdir()
    score_root.mkdir()
    panel_paths: dict[str, Path] = {}
    manifests = []
    correctness = {
        "weighted_spectral_entropy": np.asarray(
            [True, False, True, False, True, False, True, False]
        ),
        "official_dreams": np.asarray(
            [False, False, True, True, False, False, True, True]
        ),
        "full23_v132": np.asarray(
            [True, True, False, False, True, True, False, False]
        ),
    }
    gaps = {
        "official_dreams": np.asarray([0.02, 0.05, 0.09, 0.13, 0.20, 0.28, 0.36, 0.46]),
        "weighted_spectral_entropy": np.asarray([0.01, 0.04, 0.08, 0.12, 0.18, 0.26, 0.34, 0.44]),
        "full23_v132": np.asarray([0.03, 0.06, 0.10, 0.14, 0.22, 0.30, 0.38, 0.48]),
    }
    for panel_index, (panel_name, directory) in enumerate(
        (("identity_disjoint", "identity"), ("formula_disjoint", "formula"))
    ):
        panel_path, manifest = write_panel(root, panel_name, 100 * panel_index)
        panel_paths[panel_name] = panel_path
        manifests.append(manifest)
        method_scores = {
            method: pair_scores(correctness[method], gaps[method]) for method in METHODS
        }
        score_matrix = np.stack([method_scores[method] for method in METHODS])
        score_filename = f"scores_{panel_name}.npz"
        np.savez_compressed(
            score_root / score_filename,
            schema_version=np.asarray("synthetic_v1"),
            N_Q=np.asarray(8, dtype=np.int64),
            N_E=np.asarray(16, dtype=np.int64),
            N_P=np.asarray(16, dtype=np.int64),
            panel_sha256=np.asarray(sha256_file(panel_path)),
            method_names=np.asarray(METHODS),
            molecule_scores=score_matrix,
        )
        table = pd.concat(
            [query_metrics(method, method_scores[method], panel_path) for method in METHODS],
            ignore_index=True,
        )
        target = evaluation_root / directory
        target.mkdir()
        table.to_csv(target / "query_metrics.csv.gz", index=False, compression="gzip")
    manifest_path = root / "manifest.csv.gz"
    pd.concat(manifests, ignore_index=True).to_csv(
        manifest_path, index=False, compression="gzip"
    )
    return argparse.Namespace(
        evaluation_root=evaluation_root,
        score_root=score_root,
        identity_panel=panel_paths["identity_disjoint"],
        formula_panel=panel_paths["formula_disjoint"],
        manifest=manifest_path,
        panels=["identity_disjoint", "formula_disjoint"],
        output=root / "output",
        bootstrap_resamples=100,
        seed=19,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="calms_correction_contract_") as temporary:
        root = Path(temporary)
        args = write_inputs(root)
        output = run(args)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["status"] == "CALMS_CORRECTION_LANDSCAPE_COMPLETE"
        assert report["allow_model_selection"] is False
        assert report["allow_routing"] is False
        assert len(report["artifacts"]["query_ledgers"]) == 4
        summary = pd.read_csv(output / "strata_summary.tsv", sep="\t")
        all_rows = summary[summary["stratum_dimension"] == "all"]
        assert len(all_rows) == 4
        identity_wse = all_rows[
            (all_rows["panel"] == "identity_disjoint")
            & (all_rows["baseline"] == "weighted_spectral_entropy")
        ].iloc[0]
        assert int(identity_wse["queries"]) == 8
        assert int(identity_wse["corrected"]) == 2
        assert int(identity_wse["introduced"]) == 2
        assert int(identity_wse["net_correct"]) == 0
        ledger = pd.read_csv(
            output
            / "query_ledger_identity_disjoint_weighted_spectral_entropy.csv.gz"
        )
        assert set(ledger["event"]) == {
            "corrected",
            "introduced",
            "retained",
            "both_wrong",
        }
        assert set(ledger["candidate_count_bin"]) == {"2_3"}
        assert set(ledger["instrument_family"]) == {"Orbitrap_FT", "TOF_qTOF"}

        score_path = args.score_root / "scores_identity_disjoint.npz"
        with np.load(score_path, allow_pickle=False) as body:
            arrays = {key: np.asarray(body[key]) for key in body.files}
        arrays["panel_sha256"] = np.asarray("0" * 64)
        np.savez_compressed(score_path, **arrays)
        bad_args = argparse.Namespace(**vars(args))
        bad_args.panels = ["identity_disjoint"]
        bad_args.output = root / "bad_output"
        try:
            run(bad_args)
        except RuntimeError as error:
            assert "panel SHA-256" in str(error)
        else:
            raise AssertionError("panel provenance mismatch was not rejected")
        assert not bad_args.output.exists()
    print("PASS: CALMS correction-landscape contracts")


if __name__ == "__main__":
    main()
