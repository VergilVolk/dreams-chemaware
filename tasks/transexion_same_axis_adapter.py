"""Score a frozen candidate axis with the official TransExION checkpoint.

TransExION is a *pair* model, not a reusable spectrum embedding.  This adapter
therefore evaluates every query/reference spectrum pair already present in a
panel and then applies the benchmark's existing max-over-reference aggregation
to obtain one score per query/molecule edge.

Two surprising behaviours are deliberately retained because they are part of
the released implementation/checkpoint contract:

* ``MSPairSet`` builds the neutral-loss branch from query-vs-query losses;
* its attention implementation registers gradient hooks during evaluation, so
  inference must run with gradients enabled (no ``torch.no_grad``).

The resulting score is an auxiliary structural-similarity comparator.  The
released model was trained on GNPS-derived data, so results on a GNPS panel are
contaminated development evidence, never independent external confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "data/external/sota_identity_methods_20261102/TransExION"
DEFAULT_CHECKPOINT = DEFAULT_REPO / "model/GNPS_MassBank.ms.model"
CHECKPOINT_BYTES = 8_403_857
CHECKPOINT_MD5 = "e2ed38093d3bcab472ab2142a4009774"
CHECKPOINT_SHA256 = "1ad288151e7990408745451d0f0892b6183814c51bd8a9f21cdd09b1fbc3295a"
EXPECTED_REPO_COMMIT = "82d810d87d60e10042dd8617e3ae514c43615fb5"
ZENODO_RECORD = "https://zenodo.org/records/8175528"


@dataclass(frozen=True)
class RefinedSpectrum:
    row: int
    precursor_mz: float
    fragment_mz: np.ndarray
    neutral_loss: np.ndarray


@dataclass(frozen=True)
class OfficialModules:
    torch: Any
    model_class: Any
    ms_object_class: Any
    compute_feature: Any
    collate: Any
    max_peak_diff: int
    defect_bins: int


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()  # noqa: S324 - reproducing the deposited checksum
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip().lower()
    return value if len(value) == 40 else None


def inspect_asset(repo: Path, checkpoint: Path) -> dict[str, object]:
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    size = checkpoint.stat().st_size
    observed_md5 = md5_file(checkpoint)
    observed_sha256 = sha256_file(checkpoint)
    if size != CHECKPOINT_BYTES:
        raise RuntimeError(
            f"TRANSEXION_CHECKPOINT_SIZE_MISMATCH: expected={CHECKPOINT_BYTES} observed={size}"
        )
    if observed_md5 != CHECKPOINT_MD5 or observed_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError(
            "TRANSEXION_CHECKPOINT_HASH_MISMATCH: "
            f"md5={observed_md5} sha256={observed_sha256}"
        )
    commit = _git_commit(repo)
    if commit is not None and commit != EXPECTED_REPO_COMMIT:
        raise RuntimeError(
            f"TRANSEXION_REPO_COMMIT_MISMATCH: expected={EXPECTED_REPO_COMMIT} observed={commit}"
        )
    return {
        "repository": str(repo.resolve()),
        "repository_commit": commit,
        "expected_repository_commit": EXPECTED_REPO_COMMIT,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_bytes": size,
        "checkpoint_md5": observed_md5,
        "checkpoint_sha256": observed_sha256,
        "zenodo_record": ZENODO_RECORD,
    }


def load_official_modules(repo: Path) -> OfficialModules:
    """Import and verify the released implementation from the pinned clone."""
    import torch

    resolved = repo.resolve()
    sys.path.insert(0, str(resolved))
    try:
        from lrp.data import (  # type: ignore
            C_MAX_PEAK_DIFF,
            C_NUM_DEFFECT_BIN,
            compute_feature_by_sim_matrix,
            mspair_collate_fn,
        )
        from lrp.model import relMSSimilarityModel  # type: ignore
        from spectrum.io import MSObject  # type: ignore
    finally:
        sys.path.pop(0)
    module_path = Path(sys.modules[relMSSimilarityModel.__module__].__file__).resolve()
    if resolved not in module_path.parents:
        raise RuntimeError(f"TRANSEXION_IMPORT_PROVENANCE_MISMATCH: {module_path}")
    return OfficialModules(
        torch=torch,
        model_class=relMSSimilarityModel,
        ms_object_class=MSObject,
        compute_feature=compute_feature_by_sim_matrix,
        collate=mspair_collate_fn,
        max_peak_diff=int(C_MAX_PEAK_DIFF),
        defect_bins=int(C_NUM_DEFFECT_BIN),
    )


def load_model(
    modules: OfficialModules, checkpoint: Path, device: str,
) -> Any:
    torch = modules.torch
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # PyTorch before weights_only
        state = torch.load(checkpoint, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError("TransExION checkpoint is not a state dictionary")
    model = modules.model_class(
        modules.max_peak_diff + 1,
        modules.defect_bins + 2,
        hidden_dim=128,
        nclasses=1,
        dropout=0.1,
    )
    model.load_state_dict(state, strict=True)
    # The released testing script converts this checkpoint to float64.
    model = model.double().to(device).eval()
    return model


def validate_panel(panel: dict[str, np.ndarray]) -> None:
    required = {"query_row", "query_ptr", "molecule_ptr", "candidate_row"}
    missing = sorted(required - set(panel))
    if missing:
        raise RuntimeError(f"TransExION panel misses required arrays: {missing}")
    query_row = np.asarray(panel["query_row"], np.int64)
    query_ptr = np.asarray(panel["query_ptr"], np.int64)
    molecule_ptr = np.asarray(panel["molecule_ptr"], np.int64)
    candidate_row = np.asarray(panel["candidate_row"], np.int64)
    if (
        query_row.ndim != 1 or candidate_row.ndim != 1
        or query_ptr.ndim != 1 or molecule_ptr.ndim != 1
        or not len(query_row) or not len(candidate_row)
        or len(query_ptr) != len(query_row) + 1
        or query_ptr[0] != 0 or query_ptr[-1] != len(molecule_ptr) - 1
        or molecule_ptr[0] != 0 or molecule_ptr[-1] != len(candidate_row)
        or np.any(np.diff(query_ptr) < 1) or np.any(np.diff(molecule_ptr) < 1)
        or np.any(query_row < 0) or np.any(candidate_row < 0)
    ):
        raise RuntimeError("TransExION panel axis/pointer contract is malformed")


def panel_rows(panel: dict[str, np.ndarray]) -> np.ndarray:
    validate_panel(panel)
    return np.unique(np.concatenate((
        np.asarray(panel["query_row"], np.int64),
        np.asarray(panel["candidate_row"], np.int64),
    ))).astype(np.int64, copy=False)


def refine_record(
    modules: OfficialModules, row: int, precursor_mz: float, peaks: np.ndarray,
) -> RefinedSpectrum:
    """Apply the released ``MSObject`` preprocessing without alterations."""
    peaks = np.asarray(peaks, dtype=np.float64)
    if peaks.ndim != 2 or peaks.shape[0] != 2 or not peaks.shape[1]:
        raise RuntimeError(f"TransExION row {row} has malformed peaks")
    if not np.isfinite(peaks).all() or not np.isfinite(precursor_mz):
        raise RuntimeError(f"TransExION row {row} contains non-finite values")
    if precursor_mz <= 0 or float(np.max(peaks[1])) <= 0:
        raise RuntimeError(f"TransExION row {row} has non-positive precursor/intensity")
    spectrum = modules.ms_object_class(
        int(row), "", "", float(precursor_mz), "",
        peaks[0].tolist(), peaks[1].tolist(),
    )
    spectrum.normalize()
    spectrum.remove_noise_peaks()
    spectrum.remove_dense_peaks()
    if not len(spectrum.mz):
        raise RuntimeError(f"TransExION row {row} has no peaks after official preprocessing")
    neutral_loss = spectrum.get_neutral_loss()
    return RefinedSpectrum(
        row=int(row),
        precursor_mz=float(precursor_mz),
        fragment_mz=np.asarray(spectrum.mz, dtype=np.float64),
        neutral_loss=np.asarray(neutral_loss, dtype=np.float64),
    )


def load_selected_spectra(
    modules: OfficialModules, mgf: Path, selected_rows: np.ndarray,
) -> dict[int, RefinedSpectrum]:
    # Preserve the benchmark's MGF block order, but retain float64 peak values
    # as in pyteomics/the released converter (the project's general encoder
    # parser casts to float32 before model-specific preprocessing).
    wanted = set(map(int, selected_rows))
    maximum = int(selected_rows[-1])
    output: dict[int, RefinedSpectrum] = {}
    for row, (_fields, precursor, peaks) in enumerate(iter_mgf_float64(mgf)):
        if row in wanted:
            output[row] = refine_record(modules, row, float(precursor), np.asarray(peaks))
        if row >= maximum:
            break
    missing = sorted(wanted - set(output))
    if missing:
        raise RuntimeError(f"MGF misses {len(missing)} selected panel rows; first={missing[:5]}")
    return output


def iter_mgf_float64(path: Path) -> Iterable[tuple[dict[str, str], float, np.ndarray]]:
    fields: dict[str, str] | None = None
    peaks: list[tuple[float, float]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                fields, peaks = {}, []
            elif line == "END IONS":
                if fields is not None and peaks and "PEPMASS" in fields:
                    try:
                        precursor = float(fields["PEPMASS"].split()[0])
                    except (ValueError, IndexError):
                        fields = None
                        continue
                    yield fields, precursor, np.asarray(peaks, dtype=np.float64).T
                fields = None
            elif fields is not None and "=" in line:
                key, value = line.split("=", 1)
                fields[key.strip().upper()] = value.strip()
            elif fields is not None:
                values = line.split()
                if len(values) >= 2:
                    try:
                        peaks.append((float(values[0]), float(values[1])))
                    except ValueError:
                        pass


def official_pair_features(
    modules: OfficialModules, query: RefinedSpectrum, reference: RefinedSpectrum,
) -> tuple[Any, Any, Any, Any, float]:
    difference = abs(query.precursor_mz - reference.precursor_mz)
    mz_feature, mz_mask = modules.compute_feature(
        query.fragment_mz, reference.fragment_mz, difference,
    )
    # Exact released MSPairSet behaviour.  Do not 'repair' reference losses.
    loss_feature, loss_mask = modules.compute_feature(
        query.neutral_loss, query.neutral_loss, difference,
    )
    return mz_feature, mz_mask, loss_feature, loss_mask, 0.0


def _pair_axes(panel: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    query_ptr = np.asarray(panel["query_ptr"], np.int64)
    molecule_ptr = np.asarray(panel["molecule_ptr"], np.int64)
    molecule_query = np.repeat(np.arange(len(query_ptr) - 1), np.diff(query_ptr))
    pair_query = np.repeat(molecule_query, np.diff(molecule_ptr))
    return pair_query, np.asarray(panel["candidate_row"], np.int64)


def score_pairs(
    modules: OfficialModules,
    model: Any,
    panel: dict[str, np.ndarray],
    spectra: dict[int, RefinedSpectrum],
    device: str,
    batch_size: int,
    progress_every: int = 0,
) -> np.ndarray:
    validate_panel(panel)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    query_rows = np.asarray(panel["query_row"], np.int64)
    pair_query, candidate_rows = _pair_axes(panel)
    result = np.empty(len(candidate_rows), dtype=np.float64)
    started = time.monotonic()
    torch = modules.torch
    # This explicitly counters an enclosing no_grad context: official attention
    # registers hooks on tensors and fails if autograd is disabled.
    with torch.enable_grad():
        for left in range(0, len(candidate_rows), batch_size):
            right = min(left + batch_size, len(candidate_rows))
            batch = []
            for index in range(left, right):
                query = spectra[int(query_rows[int(pair_query[index])])]
                reference = spectra[int(candidate_rows[index])]
                batch.append(official_pair_features(modules, query, reference))
            mz, mz_mask, loss, loss_mask, _target = modules.collate(batch)
            output = model(
                mz.to(device), mz_mask.to(device),
                loss.to(device), loss_mask.to(device),
            )
            values = output.reshape(-1).detach().cpu().numpy().astype(np.float64, copy=False)
            if len(values) != right - left or not np.all(np.isfinite(values)):
                raise RuntimeError("TransExION produced malformed or non-finite pair scores")
            result[left:right] = values
            if progress_every and (right == len(candidate_rows) or right // progress_every != left // progress_every):
                elapsed = max(time.monotonic() - started, 1e-9)
                print(json.dumps({
                    "status": "TRANSEXION_PAIR_SCORE_PROGRESS",
                    "pairs": right,
                    "total_pairs": len(candidate_rows),
                    "pairs_per_second": round(right / elapsed, 3),
                }), flush=True)
    return result


def aggregate_molecules(pair_scores: np.ndarray, molecule_ptr: np.ndarray) -> np.ndarray:
    pair_scores = np.asarray(pair_scores, np.float64)
    molecule_ptr = np.asarray(molecule_ptr, np.int64)
    if (
        pair_scores.ndim != 1 or molecule_ptr.ndim != 1
        or not len(pair_scores) or len(molecule_ptr) < 2
        or molecule_ptr[0] != 0 or molecule_ptr[-1] != len(pair_scores)
        or np.any(np.diff(molecule_ptr) < 1) or not np.all(np.isfinite(pair_scores))
    ):
        raise RuntimeError("TransExION pair scores cannot be aggregated on this molecule axis")
    output = np.maximum.reduceat(pair_scores, molecule_ptr[:-1]).astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("TransExION molecule aggregation produced non-finite scores")
    return output


def write_score_cache(
    output: Path,
    panel_path: Path,
    panel: dict[str, np.ndarray],
    method_name: str,
    molecule_scores: np.ndarray,
) -> None:
    n_q = len(panel["query_row"])
    n_e = len(panel["molecule_ptr"]) - 1
    n_p = len(panel["candidate_row"])
    if molecule_scores.shape != (n_e,):
        raise RuntimeError(f"TransExION molecule score shape mismatch: {molecule_scores.shape}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray("mega_identity_method_scores_v1"),
        panel_sha256=np.asarray(sha256_file(panel_path)),
        N_Q=np.asarray(n_q, np.int64),
        N_E=np.asarray(n_e, np.int64),
        N_P=np.asarray(n_p, np.int64),
        method_names=np.asarray([method_name]),
        molecule_scores=molecule_scores[None, :],
    )


def _write_json(path: Path, body: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")


def score_command(args: argparse.Namespace) -> None:
    panel_path = args.panel
    mgf_path = args.mgf
    if args.benchmark is not None:
        if panel_path is None:
            if not args.panel_name:
                raise RuntimeError("--benchmark requires --panel-name or an explicit --panel")
            panel_path = args.benchmark / f"panel_{args.panel_name}.npz"
        if mgf_path is None:
            mgf_path = args.benchmark / "spectra.mgf"
    if panel_path is None or not panel_path.is_file():
        raise FileNotFoundError(panel_path)
    if mgf_path is None or not mgf_path.is_file():
        raise FileNotFoundError(mgf_path)
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite a TransExION score cache or report")
    asset = inspect_asset(args.repo.resolve(), args.checkpoint.resolve())
    modules = load_official_modules(args.repo.resolve())
    torch = modules.torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "cpu" and args.cpu_threads is not None:
        if args.cpu_threads < 1:
            raise ValueError("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)
    with np.load(panel_path, allow_pickle=False) as body:
        panel = {name: np.asarray(body[name]) for name in body.files}
    validate_panel(panel)
    selected = panel_rows(panel)
    spectra = load_selected_spectra(modules, mgf_path, selected)
    model = load_model(modules, args.checkpoint.resolve(), args.device)
    pair_scores = score_pairs(
        modules, model, panel, spectra, args.device,
        args.batch_size, args.progress_every,
    )
    molecule_scores = aggregate_molecules(pair_scores, panel["molecule_ptr"])
    write_score_cache(args.output, panel_path, panel, args.method_name, molecule_scores)
    report = {
        "schema": "transexion_same_axis_pair_score_v1",
        "status": "complete",
        "method_name": args.method_name,
        "method_role": "auxiliary_structural_similarity_comparator",
        "exact_identity_specific_objective": False,
        "asset": asset,
        "panel": str(panel_path.resolve()),
        "panel_sha256": sha256_file(panel_path),
        "mgf": str(mgf_path.resolve()),
        "mgf_sha256": sha256_file(mgf_path),
        "counts": {
            "selected_spectrum_rows": int(len(selected)),
            "N_Q": int(len(panel["query_row"])),
            "N_E": int(len(molecule_scores)),
            "N_P": int(len(pair_scores)),
        },
        "scoring": {
            "pair_score": "raw released TransExION regression output; no clipping or calibration",
            "molecule_aggregation": "maximum over frozen candidate reference spectra",
            "model_dtype": "float64, matching released test script",
            "batch_size": int(args.batch_size),
            "device": args.device,
            "gradients_enabled_during_forward": True,
            "neutral_loss_branch": "released MSPairSet query-vs-query behaviour retained",
            "preprocessing": (
                "released MSObject normalize; relative intensity >=0.001; 10<=m/z<=1000; "
                "released dense-peak removal; neutral losses from final retained fragment"
            ),
        },
        "contamination_boundary": {
            "checkpoint_training_source": "GNPS-derived spectra (released GNPS_MassBank asset)",
            "gnps_panel_status": "known source-domain overlap; exact identity overlap unquantified",
            "allowed_gnps_claim": "consumed/development auxiliary comparison only",
            "forbidden_claim": "independent external confirmation or exact-identity-specific SOTA",
        },
        "score_range": {
            "pair_min": float(np.min(pair_scores)),
            "pair_max": float(np.max(pair_scores)),
            "molecule_min": float(np.min(molecule_scores)),
            "molecule_max": float(np.max(molecule_scores)),
        },
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    _write_json(args.report, report)
    print(json.dumps({
        "status": "TRANSEXION_SAME_AXIS_PAIR_SCORE_COMPLETE",
        "output": str(args.output),
        "N_Q": report["counts"]["N_Q"],
        "N_E": report["counts"]["N_E"],
        "N_P": report["counts"]["N_P"],
    }), flush=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--benchmark", type=Path)
    score.add_argument("--mgf", type=Path)
    score.add_argument("--panel", type=Path)
    score.add_argument("--panel-name")
    score.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    score.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    score.add_argument("--device", default="cpu")
    score.add_argument("--cpu-threads", type=int)
    score.add_argument("--batch-size", type=int, default=32)
    score.add_argument("--progress-every", type=int, default=10_000)
    score.add_argument("--method-name", default="TransExION (GNPS-trained; auxiliary)")
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--report", type=Path, required=True)
    score.set_defaults(function=score_command)
    return root


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
