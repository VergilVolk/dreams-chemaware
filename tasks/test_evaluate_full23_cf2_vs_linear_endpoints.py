#!/usr/bin/env python
"""Synthetic contracts for the read-only CF2-versus-linear evaluator."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np

from evaluate_full23_cf2_vs_linear_endpoints import (
    CF2_METHOD,
    FROZEN_AXIS_SCORES,
    LINEAR_METHOD,
    PARENT,
    SEQUENTIAL_PARENT,
    WSE,
    evaluate,
)


PANELS = ("identity_disjoint", "formula_disjoint")


def _frozen_scores(offset: float) -> dict[str, np.ndarray]:
    base = np.asarray([0.9, 0.2, 0.1, 0.8, 0.7, 0.3], dtype=np.float32)
    return {
        name: base + offset + index * 0.013
        for index, name in enumerate(FROZEN_AXIS_SCORES)
    }


def _write_run(
    path: Path, method: str, method_scores: dict[tuple[str, str], np.ndarray],
    corrupt_frozen_axis: bool = False,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for source_index, source in enumerate(("enveda", "gnps")):
        for panel_index, panel in enumerate(PANELS):
            prefix = f"{source}__{panel}__"
            query_keys = (
                np.asarray(["ev-q0", "ev-q1"], dtype="U")
                if source == "enveda"
                else np.asarray(["row=10", "row=11"], dtype="U")
            )
            arrays.update({
                prefix + "query_ptr": np.asarray([0, 3, 6], dtype=np.int64),
                prefix + "labels": np.asarray([1, 0, 0, 0, 1, 0], dtype=np.int8),
                prefix + "query_keys": query_keys,
                prefix + "formula": np.asarray(["C6H12O6", "C7H14O7"], dtype="U"),
                prefix + "near": np.asarray([False, True]),
                prefix + method: method_scores[(source, panel)],
            })
            frozen = _frozen_scores(source_index * 0.1 + panel_index * 0.01)
            if corrupt_frozen_axis and source == "enveda" and panel == "identity_disjoint":
                frozen["noise_v1"] = frozen["noise_v1"].copy()
                frozen["noise_v1"][[0, 1]] = frozen["noise_v1"][[1, 0]]
            arrays.update({prefix + name: value for name, value in frozen.items()})
    np.savez_compressed(path / "candidate_predictions_outer_oof.npz", **arrays)
    (path / "report.json").write_text(json.dumps({
        "status": f"SYNTHETIC_{method.upper()}_COMPLETE",
        "method": method,
    }), encoding="utf-8")


def _write_external(root: Path) -> dict[str, Path]:
    enveda_arrays: dict[str, np.ndarray] = {}
    score_dir = root / "gnps_scores"
    score_dir.mkdir()
    panel_paths = {}
    axis_arrays: dict[str, np.ndarray] = {}
    for panel_index, panel in enumerate(PANELS):
        prefix = panel + "__"
        frozen_enveda = _frozen_scores(panel_index * 0.01)
        enveda_arrays.update({
            prefix + "query_ptr": np.asarray([0, 3, 6], dtype=np.int64),
            prefix + "labels": np.asarray([1, 0, 0, 0, 1, 0], dtype=np.int8),
            prefix + "query_keys": np.asarray(["ev-q0", "ev-q1"], dtype="U"),
            prefix + "formula": np.asarray(["C6H12O6", "C7H14O7"], dtype="U"),
            prefix + "near": np.asarray([False, True]),
            prefix + "additive_linear_control": frozen_enveda[PARENT],
            prefix + WSE: frozen_enveda[WSE],
            prefix + "official_dreams": frozen_enveda["official_dreams"],
            prefix + "noise_v1": frozen_enveda["noise_v1"],
            prefix + "p2b_noise_v1_frozen": frozen_enveda["p2b_noise_v1_frozen"],
        })

        frozen_gnps = _frozen_scores(0.1 + panel_index * 0.01)
        np.savez_compressed(
            score_dir / f"candidate_scores_{panel}.npz",
            query_ptr=np.asarray([0, 3, 6], dtype=np.int64),
            molecule_label=np.asarray([1, 0, 0, 0, 1, 0], dtype=np.int8),
            query_key=np.asarray(["row=10", "row=11"], dtype="U"),
            additive_linear_control=frozen_gnps[PARENT],
            weighted_spectral_entropy=frozen_gnps[WSE],
            official_dreams=frozen_gnps["official_dreams"],
            noise_v1=frozen_gnps["noise_v1"],
            p2b_noise_v1_frozen=frozen_gnps["p2b_noise_v1_frozen"],
        )
        axis_prefix = "identity__" if panel == "identity_disjoint" else "formula__"
        axis_arrays.update({
            axis_prefix + "near": np.asarray([False, True]),
            axis_prefix + "formula": np.asarray(["C6H12O6", "C7H14O7"], dtype="U"),
            axis_prefix + "query_ids": np.asarray(
                ["ABCDEFGHIJKLMN", "OPQRSTUVWXYZAB"], dtype="U"
            ),
        })
        panel_path = root / f"panel_{panel}.npz"
        np.savez_compressed(
            panel_path,
            query_row=np.asarray([10, 11], dtype=np.int64),
            query_ik14=np.asarray(["ABCDEFGHIJKLMN", "OPQRSTUVWXYZAB"], dtype="U"),
            query_formula=np.asarray(["C6H12O6", "C7H14O7"], dtype="U"),
            near_query=np.asarray([False, True]),
            query_ptr=np.asarray([0, 3, 6], dtype=np.int64),
            molecule_label=np.asarray([1, 0, 0, 0, 1, 0], dtype=np.int8),
        )
        panel_paths[panel] = panel_path
    enveda = root / "enveda_ledger.npz"
    np.savez_compressed(enveda, **enveda_arrays)
    axis = root / "gnps_axis.npz"
    np.savez_compressed(axis, **axis_arrays)
    return {
        "enveda": enveda,
        "score_dir": score_dir,
        "axis": axis,
        **panel_paths,
    }


def _args(root: Path, cf2: Path, linear: Path, output: Path, external: dict) -> SimpleNamespace:
    return SimpleNamespace(
        candidate_run=cf2, baseline_run=linear,
        candidate_method=None, baseline_method=None,
        enveda_ledger=external["enveda"],
        gnps_score_dir=external["score_dir"],
        gnps_axis_ledger=external["axis"],
        gnps_identity_panel=external["identity_disjoint"],
        gnps_formula_panel=external["formula_disjoint"],
        output=output, bootstrap_resamples=100, seed=19,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cf2_vs_linear_contract_") as tmp:
        root = Path(tmp)
        cf2, linear, corrupted = root / "cf2", root / "linear", root / "corrupted"
        cf2.mkdir(); linear.mkdir(); corrupted.mkdir()
        external = _write_external(root)
        cf2_scores = {
            (source, panel): np.asarray([0.95, 0.1, 0.2, 0.2, 0.9, 0.1], np.float32)
            for source in ("enveda", "gnps") for panel in PANELS
        }
        linear_scores = {
            (source, panel): np.asarray([0.2, 0.9, 0.1, 0.2, 0.8, 0.1], np.float32)
            for source in ("enveda", "gnps") for panel in PANELS
        }
        _write_run(cf2, CF2_METHOD, cf2_scores)
        _write_run(linear, LINEAR_METHOD, linear_scores)
        report = evaluate(_args(root, cf2, linear, root / "result", external))
        assert report["status"] == "FULL23_CF2_VS_LINEAR_ENDPOINT_EVALUATION_COMPLETE"
        assert report["panels"]["enveda"]["identity_disjoint"][
            f"{CF2_METHOD}_vs_{LINEAR_METHOD}"
        ]["corrected"] == 1
        assert report["query_balanced_primary_endpoints"]["gnps"][
            f"{CF2_METHOD}_vs_{LINEAR_METHOD}"
        ]["top1_delta"] > 0
        assert report["gnps_identity_ik14_collapse_sensitivity"]["unique_units"] == 2

        # Same lengths and method scores are insufficient: changing one frozen
        # candidate-axis fingerprint must fail closed before endpoint scoring.
        _write_run(corrupted, CF2_METHOD, cf2_scores, corrupt_frozen_axis=True)
        try:
            evaluate(_args(root, corrupted, linear, root / "must_not_exist", external))
        except RuntimeError as error:
            assert "query/candidate axis mismatch" in str(error)
        else:
            raise AssertionError("candidate-axis corruption was accepted")
    print("PASS: Full23 CF2-vs-linear endpoint evaluator contracts")


if __name__ == "__main__":
    main()
