#!/usr/bin/env python3
"""Build one phenotype-blind, MS2-anchored Pan-ReDU abundance matrix.

Anchors are recurrent precursor/RT events supported by distinct biological
units.  Representative MS2 spectra and per-file MS1 peak heights are persisted
with resumable caches.  The program deliberately has no contrast membership
input; effects are estimated only after this output is frozen.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms
from ms_entropy import clean_spectrum
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN


BANNED_COLUMNS = {
    "contrast_id", "contrast_field", "level", "level_a", "level_b",
    "group", "case_control", "outcome", "phenotype",
}
_QUANT_MZ: np.ndarray | None = None
_QUANT_RT: np.ndarray | None = None
_QUANT_PPM = 0.0
_QUANT_WINDOW = 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def cache_path(root: Path, sample_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id)
    return root / f"{safe}.npz"


def load_experiment(path: Path) -> oms.MSExperiment:
    experiment = oms.MSExperiment()
    options = oms.PeakFileOptions()
    options.setMSLevels([1, 2])
    suffix = path.suffix.casefold()
    if suffix == ".mzml":
        loader = oms.MzMLFile()
    elif suffix == ".mzxml":
        loader = oms.MzXMLFile()
    else:
        raise ValueError(f"unsupported raw format: {path}")
    loader.setOptions(options)
    loader.load(str(path), experiment)
    return experiment


def valid_envelope(path: Path, expected_bytes: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_bytes:
        raise IOError(f"byte mismatch {path}: {path.stat().st_size} != {expected_bytes}")
    with path.open("rb") as handle:
        head = handle.read(4096).lower()
        handle.seek(max(0, expected_bytes - 8192))
        tail = handle.read().lower()
    if path.suffix.casefold() == ".mzml":
        valid = b"mzml" in head and (b"</mzml>" in tail or b"</indexedmzml>" in tail)
    elif path.suffix.casefold() == ".mzxml":
        valid = b"mzxml" in head and b"</mzxml>" in tail
    else:
        valid = False
    if not valid:
        raise IOError(f"invalid mzML/mzXML envelope: {path}")


def inventory_worker(task: tuple[str, str, str, int, float]) -> dict[str, object]:
    sample_id, raw_name, cache_name, expected_bytes, fragment_da = task
    path, saved = Path(raw_name), Path(cache_name)
    valid_envelope(path, expected_bytes)
    if saved.is_file():
        with np.load(saved, allow_pickle=False) as cached:
            if int(cached["input_bytes"][0]) != expected_bytes:
                raise RuntimeError(f"stale inventory cache: {saved}")
            return {
                "sample_id": sample_id,
                "eligible_ms2": int(len(cached["precursor_mz"])),
                "sha256": str(cached["input_sha256"][0]),
                "status": "reused",
            }
    experiment = load_experiment(path)
    precursor_mz: list[float] = []
    rt_seconds: list[float] = []
    native_id: list[str] = []
    tic: list[float] = []
    offsets = [0]
    peak_mz: list[np.ndarray] = []
    peak_intensity: list[np.ndarray] = []
    for spectrum in experiment:
        if spectrum.getMSLevel() != 2 or not spectrum.getPrecursors():
            continue
        precursor = float(spectrum.getPrecursors()[0].getMZ())
        rt = float(spectrum.getRT())
        mz, intensity = spectrum.get_peaks()
        if not np.isfinite(precursor) or precursor <= 0 or not np.isfinite(rt):
            continue
        raw = np.column_stack((mz, intensity)).astype(np.float32, copy=False)
        cleaned = np.asarray(clean_spectrum(
            raw,
            noise_threshold=0.01,
            min_ms2_difference_in_da=fragment_da,
            max_peak_num=100,
            normalize_intensity=True,
        ), dtype=np.float32)
        if len(cleaned) < 6:
            continue
        precursor_mz.append(precursor)
        rt_seconds.append(rt)
        native_id.append(str(spectrum.getNativeID()))
        tic.append(float(np.sum(intensity)))
        peak_mz.append(cleaned[:, 0])
        peak_intensity.append(cleaned[:, 1])
        offsets.append(offsets[-1] + len(cleaned))
    flat_mz = np.concatenate(peak_mz).astype(np.float32) if peak_mz else np.empty(0, dtype=np.float32)
    flat_intensity = np.concatenate(peak_intensity).astype(np.float32) if peak_intensity else np.empty(0, dtype=np.float32)
    input_digest = sha256(path)
    saved.parent.mkdir(parents=True, exist_ok=True)
    temporary = saved.with_suffix(saved.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            precursor_mz=np.asarray(precursor_mz, dtype=np.float64),
            rt_seconds=np.asarray(rt_seconds, dtype=np.float64),
            native_id=np.asarray(native_id, dtype=str),
            tic=np.asarray(tic, dtype=np.float64),
            peak_offsets=np.asarray(offsets, dtype=np.int64),
            peak_mz=flat_mz,
            peak_intensity=flat_intensity,
            input_bytes=np.asarray([expected_bytes], dtype=np.int64),
            input_sha256=np.asarray([input_digest], dtype=str),
        )
    temporary.replace(saved)
    return {"sample_id": sample_id, "eligible_ms2": len(precursor_mz), "sha256": input_digest, "status": "built"}


def compact_observations(manifest: pd.DataFrame, inventory_dir: Path) -> dict[str, np.ndarray]:
    unit_codes = {value: index for index, value in enumerate(sorted(manifest["biological_unit_id"].unique()))}
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in ("mz", "rt", "tic", "file", "unit", "local")}
    for file_index, row in enumerate(manifest.itertuples(index=False)):
        with np.load(cache_path(inventory_dir, row.sample_id), allow_pickle=False) as cached:
            count = len(cached["precursor_mz"])
            pieces["mz"].append(np.asarray(cached["precursor_mz"], dtype=np.float64))
            pieces["rt"].append(np.asarray(cached["rt_seconds"], dtype=np.float64))
            pieces["tic"].append(np.asarray(cached["tic"], dtype=np.float64))
            pieces["file"].append(np.full(count, file_index, dtype=np.int32))
            pieces["unit"].append(np.full(count, unit_codes[row.biological_unit_id], dtype=np.int32))
            pieces["local"].append(np.arange(count, dtype=np.int32))
    if not pieces["mz"] or sum(len(value) for value in pieces["mz"]) == 0:
        raise RuntimeError("no eligible MS2 observations")
    return {key: np.concatenate(value) for key, value in pieces.items()}


def cluster_anchors(
    obs: dict[str, np.ndarray], minimum_units: int, precursor_ppm: float, rt_seconds: float,
) -> pd.DataFrame:
    coordinates = np.column_stack((np.log(obs["mz"]) / (precursor_ppm * 1e-6), obs["rt"] / rt_seconds))
    connected = DBSCAN(
        eps=1.0, min_samples=1, metric="chebyshev", algorithm="kd_tree", n_jobs=-1,
    ).fit_predict(coordinates)
    rows: list[dict[str, object]] = []
    for component in np.unique(connected):
        component_indices = np.flatnonzero(connected == component)
        if len(component_indices) == 1:
            subgroups = [component_indices]
        else:
            hierarchy = linkage(coordinates[component_indices], method="complete", metric="chebyshev")
            sublabels = fcluster(hierarchy, t=1.0, criterion="distance")
            subgroups = [component_indices[sublabels == value] for value in np.unique(sublabels)]
        for indices in subgroups:
            units = np.unique(obs["unit"][indices])
            if len(units) < minimum_units:
                continue
            representative = int(indices[np.argmax(obs["tic"][indices])])
            rows.append({
                "precursor_mz": float(np.median(obs["mz"][indices])),
                "rt_seconds": float(np.median(obs["rt"][indices])),
                "supporting_biological_units": int(len(units)),
                "supporting_files": int(len(np.unique(obs["file"][indices]))),
                "ms2_observations": int(len(indices)),
                "representative_file_index": int(obs["file"][representative]),
                "representative_local_index": int(obs["local"][representative]),
                "representative_tic": float(obs["tic"][representative]),
                "cluster_mz_span_ppm": float((np.max(obs["mz"][indices]) / np.min(obs["mz"][indices]) - 1.0) * 1e6),
                "cluster_rt_span_seconds": float(np.max(obs["rt"][indices]) - np.min(obs["rt"][indices])),
            })
    if not rows:
        return pd.DataFrame()
    anchors = pd.DataFrame(rows).sort_values(["precursor_mz", "rt_seconds"]).reset_index(drop=True)
    anchors.insert(0, "anchor_id", [f"A{index + 1:07d}" for index in range(len(anchors))])
    return anchors


def representative_spectra(
    anchors: pd.DataFrame, manifest: pd.DataFrame, inventory_dir: Path,
) -> tuple[list[np.ndarray], list[str]]:
    spectra: list[np.ndarray | None] = [None] * len(anchors)
    native_ids = [""] * len(anchors)
    for file_index, group in anchors.groupby("representative_file_index"):
        sample_id = str(manifest.iloc[int(file_index)]["sample_id"])
        with np.load(cache_path(inventory_dir, sample_id), allow_pickle=False) as cached:
            offsets = np.asarray(cached["peak_offsets"], dtype=np.int64)
            mz = np.asarray(cached["peak_mz"], dtype=np.float32)
            intensity = np.asarray(cached["peak_intensity"], dtype=np.float32)
            native = np.asarray(cached["native_id"], dtype=str)
            for anchor_index, row in group.iterrows():
                local = int(row["representative_local_index"])
                left, right = int(offsets[local]), int(offsets[local + 1])
                spectra[int(anchor_index)] = np.column_stack((mz[left:right], intensity[left:right]))
                native_ids[int(anchor_index)] = str(native[local])
    if any(value is None for value in spectra):
        raise RuntimeError("failed to recover every representative spectrum")
    return [value for value in spectra if value is not None], native_ids


def write_mgf(path: Path, anchors: pd.DataFrame, spectra: list[np.ndarray], manifest: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row, peaks in zip(anchors.itertuples(index=False), spectra):
            source = manifest.iloc[int(row.representative_file_index)]
            handle.write("BEGIN IONS\n")
            handle.write(f"TITLE={row.anchor_id}\nFEATURE_ID={row.anchor_id}\n")
            handle.write(f"PEPMASS={row.precursor_mz:.8f}\nRTINSECONDS={row.rt_seconds:.4f}\n")
            handle.write(f"SOURCE_SAMPLE_ID={source.sample_id}\nSOURCE_NATIVE_ID={row.representative_native_id}\n")
            for mz, intensity in peaks:
                handle.write(f"{float(mz):.6f} {float(intensity):.8f}\n")
            handle.write("END IONS\n\n")


def quantify_file(path: Path, anchor_mz: np.ndarray, anchor_rt: np.ndarray,
                  precursor_ppm: float, rt_window_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    experiment = load_experiment(path)
    ms1 = [spectrum for spectrum in experiment if spectrum.getMSLevel() == 1]
    if len(ms1) < 10:
        raise RuntimeError(f"too few MS1 scans in {path}: {len(ms1)}")
    scan_rt = np.asarray([spectrum.getRT() for spectrum in ms1], dtype=float)
    if np.any(np.diff(scan_rt) < 0):
        raise RuntimeError(f"MS1 scans are not RT-sorted: {path}")
    output = np.zeros(len(anchor_mz), dtype=np.float32)
    apex_rt = np.full(len(anchor_mz), np.nan, dtype=np.float32)
    for index, (target_mz, target_rt) in enumerate(zip(anchor_mz, anchor_rt)):
        left_scan = int(np.searchsorted(scan_rt, target_rt - rt_window_seconds, side="left"))
        right_scan = int(np.searchsorted(scan_rt, target_rt + rt_window_seconds, side="right"))
        tolerance = target_mz * precursor_ppm * 1e-6
        maximum = 0.0
        for spectrum in ms1[left_scan:right_scan]:
            mz, intensity = spectrum.get_peaks()
            left = int(np.searchsorted(mz, target_mz - tolerance, side="left"))
            right = int(np.searchsorted(mz, target_mz + tolerance, side="right"))
            if right > left:
                value = float(np.sum(intensity[left:right]))
                if value > maximum:
                    maximum = value
                    apex_rt[index] = float(spectrum.getRT())
        output[index] = maximum
    return output, apex_rt


def init_quant(mz: np.ndarray, rt: np.ndarray, ppm: float, window: float) -> None:
    global _QUANT_MZ, _QUANT_RT, _QUANT_PPM, _QUANT_WINDOW
    _QUANT_MZ, _QUANT_RT, _QUANT_PPM, _QUANT_WINDOW = mz, rt, ppm, window


def quant_worker(task: tuple[str, str, str, int]) -> dict[str, object]:
    sample_id, raw_name, cache_name, expected_bytes = task
    path, saved = Path(raw_name), Path(cache_name)
    valid_envelope(path, expected_bytes)
    if saved.is_file():
        with np.load(saved, allow_pickle=False) as cached:
            if _QUANT_MZ is None or len(cached["intensity"]) != len(_QUANT_MZ):
                raise RuntimeError(f"stale quantification cache: {saved}")
        return {"sample_id": sample_id, "status": "reused"}
    assert _QUANT_MZ is not None and _QUANT_RT is not None
    intensity, apex_rt = quantify_file(path, _QUANT_MZ, _QUANT_RT, _QUANT_PPM, _QUANT_WINDOW)
    temporary = saved.with_suffix(saved.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, intensity=intensity, apex_rt=apex_rt)
    temporary.replace(saved)
    return {"sample_id": sample_id, "status": "built"}


def parallel_run(function, tasks: list[tuple], workers: int, initializer=None, initargs=()) -> list[dict[str, object]]:
    if workers == 1:
        if initializer is not None:
            initializer(*initargs)
        return [function(task) for task in tasks]
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn"), initializer=initializer, initargs=initargs,
    ) as pool:
        futures = {pool.submit(function, task): task[0] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minimum-biological-units", type=int, default=3)
    parser.add_argument("--minimum-anchors", type=int, default=25)
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    parser.add_argument("--fragment-da", type=float, default=0.02)
    parser.add_argument("--cluster-rt-seconds", type=float, default=8.0)
    parser.add_argument("--quant-rt-seconds", type=float, default=15.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str, keep_default_na=False)
    required = {"sample_id", "biological_unit_id", "local_path", "deposited_bytes"}
    if missing := sorted(required - set(manifest.columns)):
        raise ValueError(f"manifest missing columns: {missing}")
    if present := sorted(BANNED_COLUMNS & set(manifest.columns)):
        raise ValueError(f"phenotype-blind manifest contains forbidden columns: {present}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("sample_id is not unique")
    manifest["deposited_bytes"] = pd.to_numeric(manifest["deposited_bytes"], errors="raise").astype("int64")
    manifest = manifest.sort_values("sample_id").reset_index(drop=True)
    contract = {
        "manifest_sha256": sha256(args.manifest),
        "samples": len(manifest),
        "biological_units": int(manifest["biological_unit_id"].nunique()),
        "minimum_biological_units": args.minimum_biological_units,
        "minimum_anchors": args.minimum_anchors,
        "precursor_ppm": args.precursor_ppm,
        "fragment_da": args.fragment_da,
        "cluster_rt_seconds": args.cluster_rt_seconds,
        "quant_rt_seconds": args.quant_rt_seconds,
    }
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite without --resume: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "build_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("contract") != contract:
            raise RuntimeError("resume contract differs from frozen build")
    else:
        state = {"stage": "inventorying", "contract": contract}
        write_json_atomic(state_path, state)

    inventory_dir = args.output_dir / "inventory_cache"
    quant_dir = args.output_dir / "quant_cache"
    inventory_dir.mkdir(exist_ok=True)
    quant_dir.mkdir(exist_ok=True)
    inventory_tasks = [
        (row.sample_id, row.local_path, str(cache_path(inventory_dir, row.sample_id)), int(row.deposited_bytes), args.fragment_da)
        for row in manifest.itertuples(index=False)
    ]
    inventory_results = parallel_run(inventory_worker, inventory_tasks, args.workers)
    integrity = pd.DataFrame(inventory_results).sort_values("sample_id")
    integrity.to_csv(args.output_dir / "input_integrity.tsv", sep="\t", index=False)
    state["stage"] = "inventory_complete"
    state["eligible_ms2_observations"] = int(integrity["eligible_ms2"].sum())
    write_json_atomic(state_path, state)

    anchor_path = args.output_dir / "anchor_manifest.tsv"
    mgf_path = args.output_dir / "representative_spectra.mgf"
    if anchor_path.is_file() != mgf_path.is_file():
        raise RuntimeError("incomplete frozen-anchor checkpoint")
    if anchor_path.is_file():
        anchors = pd.read_csv(anchor_path, sep="\t")
    else:
        obs = compact_observations(manifest, inventory_dir)
        anchors = cluster_anchors(obs, args.minimum_biological_units, args.precursor_ppm, args.cluster_rt_seconds)
        if len(anchors) < args.minimum_anchors:
            raise RuntimeError(f"too few recurrent anchors: {len(anchors)} < {args.minimum_anchors}")
        spectra, native_ids = representative_spectra(anchors, manifest, inventory_dir)
        anchors["representative_native_id"] = native_ids
        anchors.to_csv(anchor_path, sep="\t", index=False)
        write_mgf(mgf_path, anchors, spectra, manifest)
        state["stage"] = "anchors_frozen"
        state["recurrent_anchors"] = len(anchors)
        write_json_atomic(state_path, state)
        del obs, spectra

    anchor_mz = anchors["precursor_mz"].to_numpy(np.float64)
    anchor_rt = anchors["rt_seconds"].to_numpy(np.float64)
    quant_tasks = [
        (row.sample_id, row.local_path, str(cache_path(quant_dir, row.sample_id)), int(row.deposited_bytes))
        for row in manifest.itertuples(index=False)
    ]
    parallel_run(
        quant_worker, quant_tasks, args.workers,
        initializer=init_quant, initargs=(anchor_mz, anchor_rt, args.precursor_ppm, args.quant_rt_seconds),
    )
    intensity_columns, apex_columns = [], []
    for row in manifest.itertuples(index=False):
        with np.load(cache_path(quant_dir, row.sample_id), allow_pickle=False) as cached:
            intensity_columns.append(np.asarray(cached["intensity"], dtype=np.float32))
            apex_columns.append(np.asarray(cached["apex_rt"], dtype=np.float32))
    intensities = np.column_stack(intensity_columns)
    apex_rts = np.column_stack(apex_columns)
    np.savez_compressed(
        args.output_dir / "ms1_peak_height.npz", values=intensities,
        anchor_id=anchors["anchor_id"].to_numpy(str), sample_id=manifest["sample_id"].to_numpy(str),
    )
    np.savez_compressed(
        args.output_dir / "ms1_apex_rt_seconds.npz", values=apex_rts,
        anchor_id=anchors["anchor_id"].to_numpy(str), sample_id=manifest["sample_id"].to_numpy(str),
    )
    manifest.to_csv(args.output_dir / "sample_manifest.tsv", sep="\t", index=False)
    detected = intensities > 0
    residual = np.abs(apex_rts - anchor_rt[:, None])
    finite_residual = residual[np.isfinite(residual)]
    report = {
        "schema": "pan_redu_ms2_anchored_panel_v1",
        "status": "development_consumed_abundance_matrix_frozen",
        "denominators": {
            "sample_files": len(manifest),
            "biological_units": int(manifest["biological_unit_id"].nunique()),
            "eligible_ms2_observations": int(integrity["eligible_ms2"].sum()),
            "recurrent_anchors": len(anchors),
            "nonzero_quantifications": int(detected.sum()),
            "total_quantifications": int(detected.size),
        },
        "measurement_qc": {
            "matrix_nonzero_fraction": float(detected.mean()),
            "median_anchor_file_detection_fraction": float(np.median(detected.mean(axis=1))),
            "apex_absolute_error_p95_seconds": float(np.quantile(finite_residual, 0.95)) if len(finite_residual) else None,
            "rt_alignment": "no outcome-informed alignment; fixed panel-local RT windows",
            "blank_filtering": "unavailable_in_cost_efficient_tranche_manifest",
        },
        "contract": contract,
        "leakage_status": "no contrast label or cross-study truth read; abundance matrix frozen before effect estimation",
        "claim_boundary": [
            "Recurrent MS2 anchors and MS1 peak heights are measurement objects, not biological responses.",
            "No blank files were acquired in this tranche, so blank-sensitive entities require downstream exclusion or reacquisition.",
            "Biological effects require design-aware unit-level estimation from the separately sealed membership table.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    state["stage"] = "complete"
    write_json_atomic(state_path, state)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
