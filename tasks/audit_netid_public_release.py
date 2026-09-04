#!/usr/bin/env python
"""Audit the public NetID release and freeze the usable DreaMS-MS2 bridge.

This script deliberately separates three claims:

1. the published NetID *outputs* can be checked against the published yeast
   manual curation;
2. the original optimizer cannot be called a public end-to-end reproduction
   because it imports IBM CPLEX and the release does not contain the complete
   pre-solution ILP state;
3. the mouse-liver targeted-MS2 workbooks are sufficient to construct a
   feature-labelled DreaMS edge cache, but not a structure-retrieval benchmark.

No model fitting or threshold selection is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STATUS = "netid_public_release_audit_complete"
ID_PATTERN = re.compile(r"\bID=(\d+)\b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _csv_gzip_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".csv.gz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _release_root(source_dir: Path, manifest: dict[str, Any]) -> Path:
    relative_paths = [
        Path(item["relative_path"])
        for item in manifest.get("required_files", {}).values()
    ]
    if not relative_paths:
        raise RuntimeError("source manifest has no required files")
    first_parts = {path.parts[0] for path in relative_paths if path.parts}
    if len(first_parts) != 1:
        raise RuntimeError(f"ambiguous release roots in source manifest: {first_parts}")
    root = source_dir / next(iter(first_parts))
    if not (root / "code" / "NetID_function.R").is_file():
        raise RuntimeError(f"NetID release root is incomplete: {root}")
    return root


def verify_source(source_dir: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = source_dir / "bioaware_netid_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "bioaware_netid_source_installed":
        raise RuntimeError("unexpected NetID source manifest status")
    for relative, record in manifest["required_files"].items():
        path = source_dir / record["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"source size changed: {relative}")
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"source hash changed: {relative}")
    return manifest, _release_root(source_dir, manifest)


def map_yeast_manual_to_author_output(
    manual: pd.DataFrame, raw: pd.DataFrame, output: pd.DataFrame
) -> pd.DataFrame:
    """Map curation rows by exact m/z+RT, then use NetID's preserved row order.

    ``manual_curate.id`` is not the row number in ``NetID_output.csv``.  The
    public output uses the cleaned raw-table row order.  Matching directly by
    the curation id silently corrupts the evaluation after the first missing
    raw-table id, so the mass/RT join is part of the frozen audit contract.
    """

    required_manual = {"id", "medMz", "medRt", "class", "Confidence", "Ground truth"}
    required_raw = {"medMz", "medRt"}
    required_output = {"peak_id", "class", "formula"}
    for name, frame, required in [
        ("manual", manual, required_manual),
        ("raw", raw, required_raw),
        ("output", output, required_output),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"{name} missing columns: {sorted(missing)}")
    if raw.duplicated(["medMz", "medRt"]).any():
        raise RuntimeError("raw m/z+RT keys are not unique")
    if output["peak_id"].tolist() != list(range(1, len(output) + 1)):
        raise RuntimeError("NetID output peak_id is not the preserved one-based row order")
    raw_rows = raw[["medMz", "medRt"]].copy()
    raw_rows["raw_row0"] = np.arange(len(raw_rows), dtype=np.int64)
    mapped = manual.merge(raw_rows, on=["medMz", "medRt"], how="inner", validate="m:1")
    mapped["netid_peak_id"] = mapped["raw_row0"] + 1
    mapped = mapped.merge(
        output[["peak_id", "class", "formula", "annotation"]],
        left_on="netid_peak_id",
        right_on="peak_id",
        how="left",
        validate="1:1",
        suffixes=("_truth", "_pred"),
    )
    if mapped["formula"].isna().any():
        raise RuntimeError("mapped yeast rows are missing NetID outputs")
    mapped["formula_correct"] = (
        mapped["Ground truth"].fillna("").astype(str)
        == mapped["formula"].fillna("").astype(str)
    )
    mapped["class_correct"] = (
        mapped["class_truth"].fillna("").astype(str).str.casefold()
        == mapped["class_pred"].fillna("").astype(str).str.casefold()
    )
    mapped["joint_correct"] = mapped["formula_correct"] & mapped["class_correct"]
    return mapped


def _safe_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def extract_mouse_liver_ms2(
    release_root: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    panel = release_root / "Mouse_liver_neg"
    raw = pd.read_csv(panel / "raw_data.csv")
    output = pd.read_csv(panel / "NetID_output.csv")
    if len(raw) != len(output):
        raise RuntimeError("mouse-liver raw/output row counts differ")
    if raw["groupId"].duplicated().any():
        raise RuntimeError("mouse-liver raw groupId is not unique")
    if output["peak_id"].tolist() != list(range(1, len(output) + 1)):
        raise RuntimeError("mouse-liver output row-order contract failed")
    row_by_group = {
        int(group): int(row0)
        for row0, group in enumerate(pd.to_numeric(raw["groupId"], errors="raise"))
    }

    records: list[dict[str, Any]] = []
    all_mz: list[np.ndarray] = []
    all_intensity: list[np.ndarray] = []
    offsets = [0]
    workbook_paths = sorted((panel / "MS2_neg_200524").glob("*.xlsx"))
    if not workbook_paths:
        raise RuntimeError("mouse-liver targeted-MS2 workbooks are absent")
    for workbook in workbook_paths:
        excel = pd.ExcelFile(workbook, engine="openpyxl")
        if "Sheet1" not in excel.sheet_names:
            raise RuntimeError(f"missing target table in {workbook}")
        metadata = pd.read_excel(excel, sheet_name="Sheet1")
        numeric_sheets = {name for name in excel.sheet_names if str(name).isdigit()}
        expected_sheets = {str(position + 2) for position in range(len(metadata))}
        if numeric_sheets != expected_sheets:
            raise RuntimeError(
                f"target/spectrum sheet mismatch in {workbook.name}: "
                f"expected={len(expected_sheets)} observed={len(numeric_sheets)}"
            )
        for position, target in metadata.iterrows():
            match = ID_PATTERN.search(_safe_string(target.get("Comment")))
            if match is None:
                raise RuntimeError(f"missing NetID ID in {workbook.name} row {position}")
            group_id = int(match.group(1))
            if group_id not in row_by_group:
                raise RuntimeError(f"unknown NetID groupId={group_id} in {workbook.name}")
            raw_row0 = row_by_group[group_id]
            raw_row = raw.iloc[raw_row0]
            predicted = output.iloc[raw_row0]
            precursor = float(target["Mass_m_z_"])
            if abs(precursor - float(raw_row["medMz"])) > 5e-4:
                raise RuntimeError(
                    f"precursor mismatch for groupId={group_id}: "
                    f"{precursor} vs {raw_row['medMz']}"
                )
            spectrum_raw = pd.read_excel(
                excel, sheet_name=str(position + 2), header=None
            )
            # The author workbooks use a one-cell ``0`` sheet when targeted
            # acquisition produced no spectrum.  That is explicit missingness,
            # not a one-peak spectrum and not an Excel parsing failure.
            if spectrum_raw.shape[1] < 2:
                spectrum = pd.DataFrame(columns=["mz", "intensity"])
                spectrum_available = False
            else:
                spectrum = spectrum_raw.iloc[:, :2].copy()
                spectrum.columns = ["mz", "intensity"]
                spectrum = spectrum.apply(pd.to_numeric, errors="coerce").dropna()
                spectrum = spectrum[
                    (spectrum["mz"] > 0) & (spectrum["intensity"] > 0)
                ]
                spectrum = spectrum.sort_values("mz", kind="mergesort")
                spectrum_available = bool(len(spectrum))
            mz = spectrum["mz"].to_numpy(dtype=np.float32)
            intensity = spectrum["intensity"].to_numpy(dtype=np.float32)
            all_mz.append(mz)
            all_intensity.append(intensity)
            offsets.append(offsets[-1] + len(mz))
            start = float(target["Start_min_"])
            end = float(target["End_min_"])
            raw_rt = float(raw_row["medRt"])
            records.append(
                {
                    "source_file": workbook.name,
                    "source_sheet": str(position + 2),
                    "feature_group_id": group_id,
                    "netid_peak_id": raw_row0 + 1,
                    "precursor_mz": precursor,
                    "raw_rt_min": raw_rt,
                    "window_start_min": start,
                    "window_end_min": end,
                    "rt_in_window": bool(start <= raw_rt <= end),
                    "collision_energy": float(target["x_N_CE"]),
                    "n_fragment_peaks": int(len(mz)),
                    "spectrum_available": spectrum_available,
                    "netid_class": _safe_string(predicted["class"]),
                    "netid_formula": _safe_string(predicted["formula"]),
                    "netid_annotation": _safe_string(predicted["annotation"]),
                }
            )
    inventory = pd.DataFrame.from_records(records)
    if inventory.empty:
        raise RuntimeError("no targeted MS2 spectra were extracted")
    packed = {
        "source_file": inventory["source_file"].to_numpy(dtype=str),
        "source_sheet": inventory["source_sheet"].to_numpy(dtype=str),
        "feature_group_id": inventory["feature_group_id"].to_numpy(dtype=np.int64),
        "netid_peak_id": inventory["netid_peak_id"].to_numpy(dtype=np.int64),
        "precursor_mz": inventory["precursor_mz"].to_numpy(dtype=np.float32),
        "raw_rt_min": inventory["raw_rt_min"].to_numpy(dtype=np.float32),
        "collision_energy": inventory["collision_energy"].to_numpy(dtype=np.float32),
        "peak_offsets": np.asarray(offsets, dtype=np.int64),
        "fragment_mz": np.concatenate(all_mz).astype(np.float32, copy=False),
        "fragment_intensity": np.concatenate(all_intensity).astype(np.float32, copy=False),
        "netid_class": inventory["netid_class"].to_numpy(dtype=str),
        "netid_formula": inventory["netid_formula"].to_numpy(dtype=str),
        "netid_annotation": inventory["netid_annotation"].to_numpy(dtype=str),
    }
    ge2 = inventory["n_fragment_peaks"] >= 2
    ge3 = inventory["n_fragment_peaks"] >= 3
    summary = {
        "workbooks": len(workbook_paths),
        "target_requests": len(inventory),
        "unique_feature_group_ids": int(inventory["feature_group_id"].nunique()),
        "duplicate_feature_spectra": int(
            len(inventory) - inventory["feature_group_id"].nunique()
        ),
        "nonempty_spectra": int((inventory["n_fragment_peaks"] > 0).sum()),
        "spectra_ge_2_peaks": int(ge2.sum()),
        "spectra_ge_3_peaks": int(ge3.sum()),
        "unique_features_ge_2_peaks": int(
            inventory.loc[ge2, "feature_group_id"].nunique()
        ),
        "unique_features_ge_3_peaks": int(
            inventory.loc[ge3, "feature_group_id"].nunique()
        ),
        "missing_placeholder_sheets": int(
            (inventory["n_fragment_peaks"] == 0).sum()
        ),
        "fragment_peaks": int(inventory["n_fragment_peaks"].sum()),
        "rt_window_matches": int(inventory["rt_in_window"].sum()),
        "known_netid_formula": int(
            (~inventory["netid_formula"].isin(["", "Unknown"])).sum()
        ),
        "known_netid_class": int(
            (~inventory["netid_class"].isin(["", "Unknown"])).sum()
        ),
        "structure_ground_truth_rows": 0,
        "structure_ground_truth_reason": (
            "Target IDs map to NetID features, but the workbooks do not provide "
            "independent structure labels; NetID assignments are predictions, not truth."
        ),
    }
    return inventory, packed, summary


def solver_audit(release_root: Path) -> dict[str, Any]:
    source = release_root / "code" / "NetID_function.R"
    run_script = release_root / "code" / "NetID_run_script.R"
    text = source.read_text(encoding="utf-8", errors="replace")
    run_text = run_script.read_text(encoding="utf-8", errors="replace")
    libraries = sorted(
        {
            match.strip().strip('"\'')
            for match in re.findall(
                r"^\s*library\(\s*([^,\n)]+)",
                text + "\n" + run_text,
                flags=re.MULTILINE,
            )
        }
    )
    pre_solution_exports = sorted(
        path.relative_to(release_root).as_posix()
        for path in release_root.rglob("*")
        if path.is_file()
        and (
            path.name in {"ilp_nodes.csv", "ilp_edges.csv", "NetID_output.RData"}
            or path.suffix.lower() == ".lp"
        )
    )
    return {
        "cplex_api_imported": "library(cplexAPI)" in text,
        "cplex_solver_called": "Run_cplex" in run_text,
        "r_libraries": libraries,
        "complete_pre_solution_ilp_state_bundled": bool(pre_solution_exports),
        "pre_solution_exports": pre_solution_exports,
        "cytoscape_files_are_post_solution_only": (
            "filter(ilp_solution > 0.01)" in run_text
        ),
        "exact_public_solver_reproduction_ready": False,
        "reason": (
            "The author pipeline imports cplexAPI/IBM CPLEX, and the release only "
            "exports post-solution Cytoscape nodes/edges rather than the complete "
            "unselected ILP variables and constraints. SciPy MILP cannot exactly "
            "replay an optimization state that was not exported."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path, default=Path("data/external/netid_v1/source")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/netid_public_release_audit_v2_20260831"),
    )
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != STATUS:
            raise RuntimeError(f"invalid existing report: {report_path}")
        if report.get("provenance", {}).get("script_sha256") != sha256(
            Path(__file__).resolve()
        ):
            raise RuntimeError("existing audit was created by a different script version")
        for name, record in report.get("artifacts", {}).items():
            path = output_dir / record["relative_path"]
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"existing artifact changed: {name}")
        print(f"[reuse] verified {report_path}", flush=True)
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest, release_root = verify_source(source_dir)
    yeast_raw_path = release_root / "Sc_neg" / "raw_data.csv"
    fdr_raw_path = release_root / "FDR_example" / "raw_data.csv"
    yeast_output_path = release_root / "Sc_neg" / "NetID_output.csv"
    manual_path = release_root / "FDR_example" / "manual_curate.csv"
    if sha256(yeast_raw_path) != sha256(fdr_raw_path):
        raise RuntimeError("FDR and Sc_neg raw tables are not identical")
    yeast_raw = pd.read_csv(yeast_raw_path)
    yeast_output = pd.read_csv(yeast_output_path)
    manual = pd.read_csv(manual_path)
    if len(yeast_raw) != len(yeast_output):
        raise RuntimeError("yeast raw/output row counts differ")
    mapped = map_yeast_manual_to_author_output(manual, yeast_raw, yeast_output)
    confident = mapped[mapped["Confidence"].eq(True)].copy()
    if len(confident) != 314:
        raise RuntimeError(f"expected 314 mapped confident curation rows, got {len(confident)}")

    yeast_audit_path = output_dir / "yeast_confident_manual_audit.csv.gz"
    _csv_gzip_atomic(yeast_audit_path, confident)
    liver_inventory, liver_packed, liver_summary = extract_mouse_liver_ms2(release_root)
    liver_inventory_path = output_dir / "mouse_liver_ms2_inventory.csv.gz"
    liver_cache_path = output_dir / "mouse_liver_ms2_spectra.npz"
    _csv_gzip_atomic(liver_inventory_path, liver_inventory)
    _npz_atomic(liver_cache_path, **liver_packed)

    manual_structure_library = pd.read_csv(
        release_root / "Mouse_liver_neg" / "manual_library.csv"
    )
    metabolite_structures = manual_structure_library[
        manual_structure_library["category"].eq("Metabolite")
        & manual_structure_library["SMILES"].notna()
    ]
    solver = solver_audit(release_root)
    formula_accuracy = float(confident["formula_correct"].mean())
    class_accuracy = float(confident["class_correct"].mean())
    joint_accuracy = float(confident["joint_correct"].mean())
    artifacts = {
        "yeast_confident_manual_audit": {
            "relative_path": yeast_audit_path.name,
            "bytes": yeast_audit_path.stat().st_size,
            "sha256": sha256(yeast_audit_path),
        },
        "mouse_liver_ms2_inventory": {
            "relative_path": liver_inventory_path.name,
            "bytes": liver_inventory_path.stat().st_size,
            "sha256": sha256(liver_inventory_path),
        },
        "mouse_liver_ms2_spectra": {
            "relative_path": liver_cache_path.name,
            "bytes": liver_cache_path.stat().st_size,
            "sha256": sha256(liver_cache_path),
        },
    }
    report: dict[str, Any] = {
        "status": STATUS,
        "formal": True,
        "source": {
            "doi": source_manifest.get("doi"),
            "archive_sha256": source_manifest.get("archive_sha256"),
            "source_manifest_sha256": sha256(
                source_dir / "bioaware_netid_source_manifest.json"
            ),
            "all_required_hashes_verified": True,
        },
        "solver_reproducibility": solver,
        "yeast_author_output_audit": {
            "endpoint": "MS1 feature class and molecular formula",
            "raw_rows": len(yeast_raw),
            "author_output_rows": len(yeast_output),
            "manual_rows": len(manual),
            "manual_rows_mapped_by_exact_mz_rt": len(mapped),
            "manual_rows_unmapped": len(manual) - len(mapped),
            "confident_manual_rows_total": int(manual["Confidence"].eq(True).sum()),
            "confident_manual_rows_mapped": len(confident),
            "formula_accuracy": formula_accuracy,
            "class_accuracy": class_accuracy,
            "joint_formula_and_class_accuracy": joint_accuracy,
            "formula_correct": int(confident["formula_correct"].sum()),
            "class_correct": int(confident["class_correct"].sum()),
            "joint_correct": int(confident["joint_correct"].sum()),
            "mapping_contract": (
                "exact medMz+medRt to Sc_neg raw row, then one-based preserved row "
                "order to NetID_output; manual id is never used as output row"
            ),
            "outcome_status": (
                "consumed public audit; not a new blind validation and not a DreaMS "
                "structure-retrieval endpoint"
            ),
        },
        "mouse_liver_targeted_ms2": {
            **liver_summary,
            "manual_library_rows": len(manual_structure_library),
            "manual_metabolites_with_smiles": len(metabolite_structures),
            "use_allowed": (
                "construct feature-level DreaMS spectral-similarity edges and compare "
                "edge behavior inside a fixed NetID-style graph"
            ),
            "use_forbidden": (
                "treat author NetID annotations as independent structure truth or "
                "claim direct DreaMS structure-ranking accuracy"
            ),
        },
        "gates": {
            "source_integrity": True,
            "author_output_rows_align": len(yeast_raw) == len(yeast_output),
            "mapped_confident_manual_rows_eq_314": len(confident) == 314,
            "author_formula_accuracy_ge_0_90": formula_accuracy >= 0.90,
            "mouse_liver_ms2_spectra_ge_3_peaks_ge_750": liver_summary[
                "spectra_ge_3_peaks"
            ]
            >= 750,
            "mouse_liver_unique_features_ge_3_peaks_ge_750": liver_summary[
                "unique_features_ge_3_peaks"
            ]
            >= 750,
            "pass_to_component_isolated_ms2_edge_stage": (
                liver_summary["spectra_ge_3_peaks"] >= 750
                and liver_summary["unique_features_ge_3_peaks"] >= 750
            ),
            "exact_author_solver_reproduction_ready": False,
            "direct_dreams_structure_benchmark_ready": False,
        },
        "next_stage_contract": {
            "baseline": "frozen published NetID output; author algorithm attribution",
            "single_changed_component": (
                "feature-feature experimental MS2 edge reliability only"
            ),
            "arms": [
                "author graph without a new DreaMS edge",
                "official DreaMS feature edge",
                "conservative author-evidence AND DreaMS intersection",
            ],
            "evaluation": (
                "component-isolated development audit with target-decoy/FDR; a new "
                "independent labelled panel is required for a performance claim"
            ),
            "embedding_claim": False,
        },
        "artifacts": artifacts,
        "provenance": {
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_limit": (
            "This audit verifies published NetID outputs and freezes a public targeted-"
            "MS2 feature cache. It is neither an exact CPLEX rerun, a new blind test, "
            "a structure-level DreaMS benchmark, nor evidence of SOTA performance."
        ),
    }
    _json_atomic(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
