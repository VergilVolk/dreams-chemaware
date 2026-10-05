#!/usr/bin/env python
"""One-time B44 MassBank evaluation of the frozen BioAware-Catalogue expert."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import sklearn
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from pilot_paired_layer_cka import preprocess_spectrum  # noqa: E402
from shared_dreams_inference import load_inference_model  # noqa: E402
from bioaware_portable_hgb import PortableBinaryHGB  # noqa: E402


FEATURES = [
    "spectral_score",
    "independent_member_count",
    "independent_member_intersection",
    "independent_log_degree_mean",
    "independent_log_degree_min",
]
TOPOLOGY_FEATURES = FEATURES[1:]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".csv.gz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(
            temporary, index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


class MassBankRows(Dataset):
    def __init__(self, path: Path, rows: np.ndarray, n_highest_peaks: int):
        self.path = str(path)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.n_highest_peaks = int(n_highest_peaks)
        self._handle = None

    def __len__(self) -> int:
        return len(self.rows)

    def _h5(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, item: int) -> torch.Tensor:
        row = int(self.rows[item])
        payload = json.loads(self._h5()["data"][row])
        if len(payload) != 7:
            raise RuntimeError(f"MassBank payload length drift at row {row}")
        _, _, mz, _, intensity, _, precursor = payload
        if len(mz) == 0 or len(mz) != len(intensity):
            raise RuntimeError(f"invalid sealed MassBank spectrum at row {row}")
        raw = np.vstack((np.asarray(mz, dtype=np.float32), np.asarray(intensity, dtype=np.float32)))
        return preprocess_spectrum(raw, float(precursor), self.n_highest_peaks)


class PreprocessedRows(Dataset):
    def __init__(self, spectra: np.ndarray):
        spectra = np.asarray(spectra, dtype=np.float32)
        if spectra.ndim != 3 or spectra.shape[2] != 2:
            raise RuntimeError(f"invalid portable spectrum-cache shape: {spectra.shape}")
        if not np.isfinite(spectra).all():
            raise RuntimeError("portable spectrum cache contains non-finite values")
        self.spectra = spectra

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, item: int) -> torch.Tensor:
        return torch.from_numpy(self.spectra[item])


def encode_rows(
    model: torch.nn.Module,
    hdf5_path: Path | None,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
    n_highest_peaks: int,
    spectrum_cache: Path | None = None,
) -> np.ndarray:
    if (hdf5_path is None) == (spectrum_cache is None):
        raise RuntimeError("provide exactly one of MassBank HDF5 or portable spectrum cache")
    if spectrum_cache is not None:
        with np.load(spectrum_cache, allow_pickle=False) as payload:
            cache_rows = payload["hdf5_rows"].astype(np.int64, copy=False)
            spectra = payload["spectra"].astype(np.float32, copy=False)
            cache_top_peaks = int(payload["n_highest_peaks"].item())
        if cache_top_peaks != int(n_highest_peaks):
            raise RuntimeError(
                f"portable cache top-peak mismatch: {cache_top_peaks} != {n_highest_peaks}"
            )
        if not np.array_equal(cache_rows, np.asarray(rows, dtype=np.int64)):
            missing = sorted(set(map(int, rows)) - set(map(int, cache_rows)))[:10]
            extra = sorted(set(map(int, cache_rows)) - set(map(int, rows)))[:10]
            raise RuntimeError(f"portable spectrum-cache row mismatch; missing={missing} extra={extra}")
        dataset: Dataset = PreprocessedRows(spectra)
    else:
        dataset = MassBankRows(hdf5_path, rows, n_highest_peaks)
    dimension = int(model.head.out_features)
    result = np.empty((len(rows), dimension), dtype=np.float32)
    loader = DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    dtype = next(model.parameters()).dtype
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            values = model(batch.to(device=device, dtype=dtype)).float().cpu().numpy()
            result[cursor:cursor + len(values)] = values
            cursor += len(values)
            if cursor % (batch_size * 20) == 0 or cursor == len(rows):
                print(f"[B44 encode] {cursor:,}/{len(rows):,}", flush=True)
    if cursor != len(rows) or not np.isfinite(result).all():
        raise RuntimeError("MassBank embedding cache is incomplete or non-finite")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("MassBank embedding contains zero vectors")
    return result / norms


def strict_order(group: pd.DataFrame, score_column: str) -> list[str]:
    # A wrong candidate precedes a tied truth, matching the locked strict-rank contract.
    return group.sort_values(
        [score_column, "is_positive", "candidate_ik14"],
        ascending=[False, True, True], kind="stable",
    )["candidate_ik14"].astype(str).tolist()


def candidate_scores(
    queries: pd.DataFrame,
    references: pd.DataFrame,
    embeddings: np.ndarray,
    row_position: dict[int, int],
    catalogue: pd.DataFrame,
) -> pd.DataFrame:
    query_lookup = queries.set_index("query_id")
    rows: list[dict] = []
    catalogue = catalogue.set_index("candidate_id")
    for query_id, local in references.groupby("query_id", sort=False):
        query = query_lookup.loc[query_id]
        qvec = embeddings[row_position[int(query["query_hdf5_row"])]]
        ref_positions = [row_position[int(row)] for row in local["reference_hdf5_row"]]
        local = local.copy()
        local["reference_similarity"] = embeddings[ref_positions] @ qvec
        for identity, molecule in local.groupby("candidate_ik14", sort=False):
            score = float(molecule["reference_similarity"].max())
            topology = (
                catalogue.loc[str(identity), TOPOLOGY_FEATURES]
                if str(identity) in catalogue.index
                else pd.Series(0.0, index=TOPOLOGY_FEATURES)
            )
            rows.append({
                "query_id": str(query_id),
                "candidate_ik14": str(identity),
                "truth_ik14": str(query["truth_ik14"]),
                "truth_formula": str(query["truth_formula"]),
                "ion_mode": str(query["ion_mode"]),
                "is_positive": bool(str(identity) == str(query["truth_ik14"])),
                "spectral_score": score,
                "reference_spectra": int(len(molecule)),
                **{column: float(topology[column]) for column in TOPOLOGY_FEATURES},
            })
    output = pd.DataFrame(rows)
    if output.duplicated(["query_id", "candidate_ik14"]).any():
        raise RuntimeError("B44 candidate molecule score is not unique")
    positives = output.groupby("query_id")["is_positive"].sum()
    if len(positives) != len(queries) or not positives.eq(1).all():
        raise RuntimeError("B44 candidate scores do not contain exactly one truth per query")
    return output


def apply_expert(
    candidate_frame: pd.DataFrame,
    model: object,
    gate: dict,
    permute_repeat: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_parts: list[pd.DataFrame] = []
    query_rows: list[dict] = []
    for query_id, original in candidate_frame.groupby("query_id", sort=False):
        group = original.copy().reset_index(drop=True)
        baseline_order = strict_order(group, "spectral_score")
        baseline = baseline_order[0]
        sorted_scores = np.sort(group["spectral_score"].to_numpy(float))[::-1]
        baseline_gap = float(sorted_scores[0] - sorted_scores[1])
        features = group[FEATURES].to_numpy(float, copy=True)
        if permute_repeat is not None:
            order = np.arange(len(group))
            rng = np.random.default_rng(stable_seed(seed, "B44-null", permute_repeat, query_id))
            shuffled = order[rng.permutation(len(order))]
            shift = 1 + stable_seed(seed, "B44-null-shift", permute_repeat, query_id) % (len(order) - 1)
            source = np.roll(shuffled, int(shift))
            permuted = features.copy()
            permuted[shuffled, 1:] = features[source, 1:]
            features = permuted
        baseline_position = int(group.index[group["candidate_ik14"].astype(str).eq(baseline)][0])
        probabilities = model.predict_proba(features - features[baseline_position])[:, 1]
        group["preference_probability"] = probabilities
        maximum = float(np.max(probabilities))
        top_positions = np.flatnonzero(np.isclose(probabilities, maximum, rtol=0, atol=1e-12))
        proposed = sorted(group.loc[top_positions, "candidate_ik14"].astype(str))[0]
        unique = len(top_positions) == 1
        intervene = bool(
            unique and proposed != baseline
            and baseline_gap <= float(gate["margin"]) + 1e-15
            and maximum >= float(gate["probability"]) - 1e-15
        )
        final_order = list(baseline_order)
        if intervene:
            final_order.remove(proposed)
            final_order.insert(0, proposed)
        truth = str(group["truth_ik14"].iloc[0])
        baseline_rank = baseline_order.index(truth) + 1
        final_rank = final_order.index(truth) + 1
        baseline_correct = baseline_rank == 1
        final_correct = final_rank == 1
        group["baseline_candidate"] = baseline
        group["proposed_candidate"] = proposed
        group["intervene"] = intervene
        group["final_candidate"] = final_order[0]
        scored_parts.append(group)
        query_rows.append({
            "query_id": str(query_id),
            "truth_ik14": truth,
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "ion_mode": str(group["ion_mode"].iloc[0]),
            "candidate_count": int(len(group)),
            "mapped_candidate_fraction": float((group["independent_member_count"] > 0).mean()),
            "truth_catalogue_member": bool(
                group.loc[group["is_positive"], "independent_member_count"].iloc[0] > 0
            ),
            "baseline_candidate": baseline,
            "proposed_candidate": proposed,
            "final_candidate": final_order[0],
            "proposal_probability": maximum,
            "proposal_unique": bool(unique),
            "baseline_gap": baseline_gap,
            "intervene": intervene,
            "baseline_rank": int(baseline_rank),
            "final_rank": int(final_rank),
            "baseline_correct": bool(baseline_correct),
            "final_correct": bool(final_correct),
            "corrected": bool(not baseline_correct and final_correct),
            "introduced": bool(baseline_correct and not final_correct),
            "delta_top1": int(final_correct) - int(baseline_correct),
        })
    return pd.concat(scored_parts, ignore_index=True), pd.DataFrame(query_rows)


def cluster_bootstrap(
    values: Iterable[float], clusters: Iterable[str], repeats: int, seed: int
) -> dict:
    frame = pd.DataFrame({"value": list(values), "cluster": list(clusters)})
    grouped = frame.groupby("cluster", sort=False)["value"].agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, len(grouped), len(grouped))
        draws[index] = sums[sample].sum() / counts[sample].sum()
    return {
        "mean": float(frame["value"].mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(grouped)),
        "resamples": int(repeats),
    }


def summarize(query_frame: pd.DataFrame, candidates: pd.DataFrame, repeats: int, seed: int) -> dict:
    final_top1 = query_frame["final_rank"].eq(1)
    baseline_top1 = query_frame["baseline_rank"].eq(1)
    denominators = query_frame["candidate_count"].to_numpy(float) - 1.0
    baseline_auc = (
        query_frame["candidate_count"].to_numpy(float)
        - query_frame["baseline_rank"].to_numpy(float)
    ) / denominators
    final_auc = (
        query_frame["candidate_count"].to_numpy(float)
        - query_frame["final_rank"].to_numpy(float)
    ) / denominators
    corrected = int(query_frame["corrected"].sum())
    introduced = int(query_frame["introduced"].sum())
    discordant = corrected + introduced
    report = {
        "queries": int(len(query_frame)),
        "formulas": int(query_frame["truth_formula"].nunique()),
        "baseline": {
            **{f"recall_at_{k}": float(query_frame["baseline_rank"].le(k).mean()) for k in (1, 2, 5, 10, 20)},
            "mrr": float(np.mean(1.0 / query_frame["baseline_rank"].to_numpy(float))),
            "macro_query_auc": float(np.mean(baseline_auc)),
        },
        "bioaware": {
            **{f"recall_at_{k}": float(query_frame["final_rank"].le(k).mean()) for k in (1, 2, 5, 10, 20)},
            "mrr": float(np.mean(1.0 / query_frame["final_rank"].to_numpy(float))),
            "macro_query_auc": float(np.mean(final_auc)),
        },
        "paired": {
            "delta_recall_at_1": float(final_top1.mean() - baseline_top1.mean()),
            "delta_mrr": float(np.mean(
                1.0 / query_frame["final_rank"].to_numpy(float)
                - 1.0 / query_frame["baseline_rank"].to_numpy(float)
            )),
            "delta_macro_query_auc": float(np.mean(final_auc - baseline_auc)),
            "corrected": corrected,
            "introduced": introduced,
            "risk_net_lambda2": corrected - 2 * introduced,
            "intervention_rate": float(query_frame["intervene"].mean()),
            "formula_cluster_top1_ci": cluster_bootstrap(
                query_frame["delta_top1"], query_frame["truth_formula"], repeats, seed
            ),
            "formula_cluster_mrr_ci": cluster_bootstrap(
                1.0 / query_frame["final_rank"].to_numpy(float)
                - 1.0 / query_frame["baseline_rank"].to_numpy(float),
                query_frame["truth_formula"], repeats, seed + 1,
            ),
            "formula_cluster_auc_ci": cluster_bootstrap(
                final_auc - baseline_auc, query_frame["truth_formula"], repeats, seed + 2,
            ),
            "mcnemar_exact_p": float(
                binomtest(min(corrected, introduced), discordant, 0.5).pvalue
                if discordant else 1.0
            ),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--massbank-hdf5", type=Path, default=ROOT / "data/massbank/massbank_full.hdf5")
    parser.add_argument("--spectrum-cache", type=Path)
    parser.add_argument("--portable-model", type=Path)
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--null-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    if args.bootstrap_resamples < 10000 or args.null_repeats != 5:
        raise ValueError("formal B44 requires 10,000 bootstrap resamples and five fixed nulls")
    use_portable_cache = args.spectrum_cache is not None
    use_portable_model = args.portable_model is not None
    if use_portable_cache != use_portable_model:
        raise RuntimeError("portable spectrum cache and portable model must be supplied together")
    required = [
        args.panel_dir / "report.json", args.panel_dir / "queries.csv.gz",
        args.panel_dir / "candidate_references.csv.gz",
        args.artifact_dir / "report.json",
        args.artifact_dir / "catalogue_lookup.csv.gz",
        args.official_checkpoint, args.architecture_checkpoint,
    ]
    required.append(args.spectrum_cache if use_portable_cache else args.massbank_hdf5)
    required.append(args.portable_model if use_portable_model else args.artifact_dir / "model.joblib")
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    panel_report = json.loads((args.panel_dir / "report.json").read_text(encoding="utf-8"))
    artifact_report = json.loads((args.artifact_dir / "report.json").read_text(encoding="utf-8"))
    if panel_report.get("pass_to_one_time_evaluation") is not True:
        raise RuntimeError("B44 panel is not sealed for evaluation")
    if artifact_report.get("status") != "bioaware_catalogue_v1_frozen":
        raise RuntimeError("BioAware artefact is not frozen catalogue v1")
    if artifact_report["contracts"].get("B44_read") is not False:
        raise RuntimeError("BioAware artefact was not demonstrably frozen before B44")
    if not use_portable_model and artifact_report["provenance"].get("scikit_learn_version") != sklearn.__version__:
        raise RuntimeError("scikit-learn version differs from the frozen artefact")
    provenance_checks = {
        "panel_queries": (panel_report["provenance"]["queries"], sha256(args.panel_dir / "queries.csv.gz")),
        "panel_candidates": (panel_report["provenance"]["candidate_references"], sha256(args.panel_dir / "candidate_references.csv.gz")),
        "catalogue_lookup": (artifact_report["provenance"]["catalogue_lookup"], sha256(args.artifact_dir / "catalogue_lookup.csv.gz")),
    }
    if use_portable_cache:
        with np.load(args.spectrum_cache, allow_pickle=False) as payload:
            source_massbank_sha256 = str(payload["source_massbank_sha256"].item())
        provenance_checks["portable_cache_source_massbank"] = (
            panel_report["provenance"]["massbank_hdf5"], source_massbank_sha256,
        )
    else:
        provenance_checks["massbank_hdf5"] = (
            panel_report["provenance"]["massbank_hdf5"], sha256(args.massbank_hdf5),
        )
    if use_portable_model:
        portable_probe = PortableBinaryHGB(args.portable_model, FEATURES)
        provenance_checks["portable_model_source_joblib"] = (
            artifact_report["provenance"]["model"], portable_probe.source_joblib_sha256,
        )
    else:
        provenance_checks["model"] = (
            artifact_report["provenance"]["model"], sha256(args.artifact_dir / "model.joblib"),
        )
    bad = [name for name, (expected, observed) in provenance_checks.items() if expected != observed]
    if bad:
        raise RuntimeError(f"B44 provenance mismatch: {bad}")

    queries = pd.read_csv(args.panel_dir / "queries.csv.gz")
    references = pd.read_csv(args.panel_dir / "candidate_references.csv.gz")
    catalogue = pd.read_csv(args.artifact_dir / "catalogue_lookup.csv.gz")
    if use_portable_model:
        model = portable_probe
    else:
        import joblib
        model = joblib.load(args.artifact_dir / "model.joblib")
    gate = artifact_report["deployment_gate"]
    all_rows = np.sort(np.unique(np.concatenate((
        queries["query_hdf5_row"].to_numpy(np.int64),
        references["reference_hdf5_row"].to_numpy(np.int64),
    ))))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    encoder, encoder_metadata = load_inference_model(
        args.official_checkpoint, args.architecture_checkpoint, device,
        args.n_highest_peaks, shared_checkpoint=None,
    )
    embeddings = encode_rows(
        encoder, None if use_portable_cache else args.massbank_hdf5, all_rows, device,
        args.batch_size, args.n_highest_peaks, args.spectrum_cache,
    )
    row_position = {int(row): index for index, row in enumerate(all_rows)}
    candidates = candidate_scores(queries, references, embeddings, row_position, catalogue)
    scored, per_query = apply_expert(candidates, model, gate, None, args.seed)
    overall = summarize(per_query, candidates, args.bootstrap_resamples, args.seed + 100)
    by_polarity = {
        polarity: summarize(
            per_query.loc[per_query["ion_mode"].eq(polarity)].copy(),
            candidates.loc[candidates["ion_mode"].eq(polarity)].copy(),
            args.bootstrap_resamples, args.seed + 200 + index,
        )
        for index, polarity in enumerate(sorted(per_query["ion_mode"].unique()))
    }
    if "NEGATIVE" not in by_polarity:
        raise RuntimeError("B44 has no negative-ion primary transfer panel")
    primary = by_polarity["NEGATIVE"]
    nulls = []
    null_query_frames = []
    for repeat in range(args.null_repeats):
        _, null_query = apply_expert(candidates, model, gate, repeat, args.seed)
        null_query_frames.append(null_query)
        negative_null = null_query.loc[null_query["ion_mode"].eq("NEGATIVE")]
        nulls.append({
            "repeat": repeat,
            "overall_delta_recall_at_1": float(null_query["delta_top1"].mean()),
            "overall_corrected": int(null_query["corrected"].sum()),
            "overall_introduced": int(null_query["introduced"].sum()),
            "negative_primary_delta_recall_at_1": float(negative_null["delta_top1"].mean()),
            "negative_primary_corrected": int(negative_null["corrected"].sum()),
            "negative_primary_introduced": int(negative_null["introduced"].sum()),
        })
    primary_real = per_query.loc[per_query["ion_mode"].eq("NEGATIVE")].reset_index(drop=True)
    primary_null_frames = [
        frame.loc[frame["ion_mode"].eq("NEGATIVE")].reset_index(drop=True)
        for frame in null_query_frames
    ]
    if any(not frame["query_id"].equals(primary_real["query_id"]) for frame in primary_null_frames):
        raise RuntimeError("B44 negative-primary null query order drift")
    real_minus_null = np.column_stack([
        primary_real["delta_top1"].to_numpy(float) - frame["delta_top1"].to_numpy(float)
        for frame in primary_null_frames
    ]).mean(axis=1)
    null_max = max(item["negative_primary_delta_recall_at_1"] for item in nulls)
    specificity_ci = cluster_bootstrap(
        real_minus_null, primary_real["truth_formula"], args.bootstrap_resamples,
        args.seed + 400,
    )
    gates = {
        "negative_primary_formula_ci_positive": bool(
            primary["paired"]["formula_cluster_top1_ci"]["ci_low"] > 0
        ),
        "negative_primary_corrected_gt_2x_introduced": bool(
            primary["paired"]["corrected"] > 2 * primary["paired"]["introduced"]
        ),
        "negative_primary_mrr_ci_nonnegative": bool(
            primary["paired"]["formula_cluster_mrr_ci"]["ci_low"] >= 0
        ),
        "negative_primary_real_beats_each_formula_stratum_null": bool(
            primary["paired"]["delta_recall_at_1"] > null_max
        ),
        "negative_primary_real_minus_null_formula_ci_positive": bool(specificity_ci["ci_low"] > 0),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = args.output_dir / "per_query.csv.gz"
    candidate_path = args.output_dir / "candidate_scores.csv.gz"
    atomic_csv_gzip(per_query_path, per_query)
    atomic_csv_gzip(candidate_path, scored)
    report = {
        "status": "bioaware_b44_massbank_one_time_evaluation_complete",
        "formal": True,
        "panel": overall,
        "primary_panel": "NEGATIVE; deployment gate was calibrated on negative-ion B42 OOF queries",
        "by_polarity": by_polarity,
        "formula_stratum_topology_nulls": nulls,
        "real_minus_mean_null_formula_cluster_ci": specificity_ci,
        "gates": gates,
        "pass_external_confirmation": bool(all(gates.values())),
        "rank_contract": "candidate molecule uses maximum reference cosine; ties count against truth; intervention moves one frozen proposal to rank 1 and preserves remaining DreaMS order",
        "auc_contract": "macro query AUC is induced by strict candidate order: (candidate_count-rank)/(candidate_count-1); BioAware moves the frozen proposal to rank 1 and preserves the remaining DreaMS order",
        "encoder": encoder_metadata,
        "provenance": {
            "panel_report": sha256(args.panel_dir / "report.json"),
            "artifact_report": sha256(args.artifact_dir / "report.json"),
            "spectrum_source": "portable_preprocessed_cache" if use_portable_cache else "massbank_hdf5",
            "spectrum_payload": sha256(args.spectrum_cache) if use_portable_cache else sha256(args.massbank_hdf5),
            "source_massbank_hdf5": panel_report["provenance"]["massbank_hdf5"],
            "model_format": "portable_numpy_hgb_v1" if use_portable_model else "joblib",
            "model_payload": sha256(args.portable_model) if use_portable_model else sha256(args.artifact_dir / "model.joblib"),
            "official_checkpoint": sha256(args.official_checkpoint),
            "architecture_checkpoint": sha256(args.architecture_checkpoint),
            "per_query": sha256(per_query_path),
            "candidate_scores": sha256(candidate_path),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": "External same-formula library retrieval of a frozen candidate-static catalogue expert. It is not sample-context propagation, phenotype inference, reaction mechanism, or shared-embedding improvement.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
