"""M1 margin-level causal audit for consensus-derived conditional peaks.

No model is trained. For each eligible query, a metadata-selected same-identity
positive and the official-DreaMS hardest strict-10ppm negative are fixed. The
query's consensus-derived conditional peaks are then removed and compared with
three count/intensity/mz-matched random deletions. The outcome is the change in
the full retrieval margin s(query, positive) - s(query, negative).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent
for search_path in (ROOT, ROOT / "tasks"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import pilot_multilevel_factor_activations as multi
from e1_checkpoint_io import official_head_state
from pilot_paired_layer_cka import preprocess_spectrum
from run_residual_pair_peak_occlusion import encode, matched_random_tokens, perturb, target_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-dir", type=Path, default=ROOT / "data/validation/g8r_directional_noise_v2_m0")
    parser.add_argument("--p3-dir", type=Path, default=ROOT / "data/validation/g8r_p3_test")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_directional_noise_v2_m1")
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL)
    parser.add_argument("--queries-per-identity", type=int, default=2)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--max-target-peaks", type=int, default=12)
    parser.add_argument("--max-mask-fraction", type=float, default=0.20)
    parser.add_argument("--token-tolerance", type=float, default=0.005)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
        for value in values
    ], dtype=object)


def stable_seed(*parts: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def parse_float_list(value: object) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=float)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    with np.load(path) as body:
        rows = body["rows"].astype(np.int64)
        embeddings = body["embeddings"].astype(np.float32)
    embeddings /= np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("official embedding cache contains duplicate rows")
    return rows, embeddings, {int(row): index for index, row in enumerate(rows)}


def choose_positive(query: pd.Series, peers: pd.DataFrame, seed: int) -> int | None:
    candidates = peers.loc[peers["row"] != query["row"]].copy()
    if candidates.empty:
        return None
    different = candidates.loc[candidates["condition"] != query["condition"]]
    if not different.empty:
        candidates = different
    candidates["stable_order"] = candidates["row"].map(
        lambda row: stable_seed(seed, "positive", int(query["row"]), int(row))
    )
    return int(candidates.sort_values("stable_order", kind="mergesort").iloc[0]["row"])


def build_mass_index(
    rows: np.ndarray, precursor: np.ndarray, adduct: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for ion in np.unique(adduct[rows]):
        subset = rows[adduct[rows] == ion]
        order = np.argsort(precursor[subset], kind="mergesort")
        output[str(ion)] = precursor[subset][order], subset[order]
    return output


def hardest_negative(
    row: int, precursor: np.ndarray, adduct: np.ndarray, ik14: np.ndarray,
    mass_index: dict[str, tuple[np.ndarray, np.ndarray]], embeddings: np.ndarray,
    embedding_index: dict[int, int], ppm: float,
) -> int | None:
    masses, candidates = mass_index[str(adduct[row])]
    tolerance = float(precursor[row]) * ppm * 1e-6
    left = int(np.searchsorted(masses, precursor[row] - tolerance, side="left"))
    right = int(np.searchsorted(masses, precursor[row] + tolerance, side="right"))
    candidates = candidates[left:right]
    candidates = np.asarray([
        candidate for candidate in candidates
        if ik14[candidate] != ik14[row] and int(candidate) in embedding_index
    ], dtype=np.int64)
    if not len(candidates):
        return None
    query_vector = embeddings[embedding_index[int(row)]]
    scores = np.asarray([embeddings[embedding_index[int(candidate)]] @ query_vector for candidate in candidates])
    # Aggregate at molecule identity, retaining its highest-scoring spectrum.
    best: dict[str, tuple[float, int]] = {}
    for candidate, score in zip(candidates, scores):
        identity = str(ik14[candidate])
        current = best.get(identity)
        if current is None or float(score) > current[0] or (float(score) == current[0] and int(candidate) < current[1]):
            best[identity] = float(score), int(candidate)
    return max(best.values(), key=lambda item: (item[0], -item[1]))[1]


def cluster_ci(frame: pd.DataFrame, cluster: str, column: str, n: int, seed: int) -> list[float] | None:
    values = frame.groupby(cluster, sort=False)[column].mean().dropna().to_numpy(float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(n, dtype=float)
    for index in range(n):
        draws[index] = rng.choice(values, len(values), replace=True).mean()
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def summarize(group: pd.DataFrame, bootstrap: int, seed: int) -> dict:
    identity_ci = cluster_ci(group, "ik14", "target_minus_random_margin_change", bootstrap, seed)
    formula_ci = cluster_ci(group, "formula", "target_minus_random_margin_change", bootstrap, seed + 10_000)
    absolute_identity_ci = cluster_ci(group, "ik14", "target_margin_change", bootstrap, seed + 20_000)
    absolute_formula_ci = cluster_ci(group, "formula", "target_margin_change", bootstrap, seed + 30_000)
    return {
        "queries": int(len(group)),
        "identities": int(group["ik14"].nunique()),
        "formulas": int(group["formula"].nunique()),
        "baseline_accuracy": float((group["baseline_margin"] > 0).mean()),
        "mean_target_margin_change": float(group["target_margin_change"].mean()),
        "mean_random_margin_change": float(group["random_margin_change"].mean()),
        "mean_target_minus_random_margin_change": float(group["target_minus_random_margin_change"].mean()),
        "median_target_minus_random_margin_change": float(group["target_minus_random_margin_change"].median()),
        "supportive_fraction": float((group["target_minus_random_margin_change"] > 0).mean()),
        "identity_cluster_specificity_95ci": identity_ci,
        "formula_cluster_specificity_95ci": formula_ci,
        "identity_cluster_absolute_change_95ci": absolute_identity_ci,
        "formula_cluster_absolute_change_95ci": absolute_formula_ci,
        "median_removed_peaks": float(group["removed_count"].median()),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    consensus_path = args.m0_dir / "source_peak_consensus.csv.gz"
    allow_path = args.p3_dir / "p3_p2_allowed_training_ik14.json"
    for path in (
        consensus_path, allow_path, args.embedding_cache, args.data,
        args.raw_checkpoint, args.official_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    consensus = pd.read_csv(consensus_path)
    consensus["row"] = consensus["row"].astype(np.int64)
    consensus["has_strict_10ppm_negative"] = as_bool(consensus["has_strict_10ppm_negative"])
    eligible = consensus.loc[
        (consensus["n_conditional_candidates"] > 0)
        & consensus["has_strict_10ppm_negative"]
        & (consensus["n_core_peaks"] >= 3)
    ].copy()
    eligible["stable_order"] = eligible.apply(
        lambda row: stable_seed(args.seed, "query", row["ik14"], int(row["row"])), axis=1,
    )
    eligible = eligible.sort_values(["ik14", "stable_order"], kind="mergesort")
    eligible = eligible.groupby("ik14", sort=False).head(args.queries_per_identity).copy()

    allow_body = json.loads(allow_path.read_text(encoding="utf-8"))
    allowed_rows = np.asarray(allow_body["real_train_primary"]["rows"], dtype=np.int64)
    embedding_rows, embeddings, embedding_index = load_embeddings(args.embedding_cache)
    missing_allowed = set(map(int, allowed_rows)) - set(map(int, embedding_rows))
    if missing_allowed:
        raise RuntimeError(f"official embedding cache misses {len(missing_allowed)} P2-allowed rows")

    with h5py.File(args.data, "r") as handle:
        ik14 = np.asarray([value[:14] for value in decode(handle["INCHIKEY"][:])], dtype=object)
        adduct = decode(handle["adduct"][:])
        precursor = np.asarray(handle["precursor_mz"][:], dtype=float)
        mass_index = build_mass_index(allowed_rows, precursor, adduct)
        selected_records = []
        grouped = {key: frame for key, frame in consensus.groupby(["ik14", "adduct"], sort=False)}
        for item in eligible.itertuples(index=False):
            query = pd.Series(item._asdict())
            peers = grouped[(str(item.ik14), str(item.adduct))]
            positive = choose_positive(query, peers, args.seed)
            negative = hardest_negative(
                int(item.row), precursor, adduct, ik14, mass_index,
                embeddings, embedding_index, args.ppm,
            )
            if positive is None or negative is None:
                continue
            selected_records.append({
                "query_row": int(item.row),
                "positive_row": int(positive),
                "negative_row": int(negative),
                "ik14": str(item.ik14),
                "formula": str(item.formula),
                "adduct": str(item.adduct),
                "query_condition": str(item.condition),
                "positive_condition": str(peers.loc[peers["row"] == positive, "condition"].iloc[0]),
                "cross_condition_positive": bool(str(item.condition) != str(peers.loc[peers["row"] == positive, "condition"].iloc[0])),
                "conditional_mz": str(item.conditional_mz),
                "core_mz": str(item.core_mz),
            })
        selected = pd.DataFrame(selected_records)
        if selected.empty:
            raise RuntimeError("no M1 query retained a positive and strict-10ppm negative")
        required_rows = np.unique(selected[["query_row", "positive_row", "negative_row"]].to_numpy(np.int64))
        spectra = {int(row): np.asarray(handle["spectrum"][int(row)], dtype=float) for row in np.unique(selected["query_row"])}
        precursor_lookup = {int(row): float(precursor[int(row)]) for row in np.unique(selected["query_row"])}
    missing_embeddings = set(map(int, required_rows)) - set(embedding_index)
    if missing_embeddings:
        raise RuntimeError(f"M1 triples miss {len(missing_embeddings)} official embeddings")

    tensors: list[torch.Tensor] = []
    metadata: list[dict] = []
    for item in selected.itertuples(index=False):
        clean = preprocess_spectrum(
            spectra[int(item.query_row)], precursor_lookup[int(item.query_row)], args.n_highest_peaks,
        )
        targets = target_tokens(clean, parse_float_list(item.conditional_mz), args.token_tolerance)
        core = target_tokens(clean, parse_float_list(item.core_mz), args.token_tolerance)
        values = clean.numpy()
        valid = np.flatnonzero(
            (np.arange(len(values)) > 0) & (values[:, 0] > 0) & (values[:, 1] > 0)
        )
        strong = valid[values[valid, 1] >= 0.20]
        # Controls must be weak non-core peaks as well; otherwise a shortage of
        # weak controls could silently turn this into weak-vs-base-peak deletion.
        excluded = set(targets.tolist()) | set(core.tolist()) | set(strong.tolist())
        control_capacity = sum(index not in excluded for index in valid)
        fraction_cap = max(1, int(np.floor(len(valid) * args.max_mask_fraction)))
        capacity = min(len(targets), args.max_target_peaks, fraction_cap, control_capacity)
        if capacity <= 0:
            continue
        order = np.argsort(values[targets, 1].astype(float))[::-1]
        targeted = targets[order[:capacity]].astype(np.int64)
        base = {
            "query_row": int(item.query_row), "positive_row": int(item.positive_row),
            "negative_row": int(item.negative_row), "ik14": str(item.ik14),
            "formula": str(item.formula), "adduct": str(item.adduct),
            "cross_condition_positive": bool(item.cross_condition_positive),
            "removed_count": int(len(targeted)), "valid_peak_count": int(len(valid)),
        }
        tensors.append(perturb(clean, targeted))
        metadata.append(base | {"condition": "targeted", "repeat": -1})
        for repeat in range(args.random_repeats):
            control = matched_random_tokens(
                clean, targeted, excluded,
                stable_seed(args.seed, item.query_row, "random", repeat),
            )
            if len(control) != len(targeted):
                continue
            tensors.append(perturb(clean, control))
            metadata.append(base | {"condition": "matched_random", "repeat": repeat})
    if not tensors:
        raise RuntimeError("no consensus interventions retained matched controls")
    print(f"[M1] queries={selected.shape[0]:,} variants={len(tensors):,}", flush=True)

    raw_package = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
    official_package = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    model = multi.reconstruct_backbone(raw_package, multi.official_backbone_state(official_package), device)
    head = official_head_state(official_package)
    weight = head["weight"].to(device=device, dtype=next(model.parameters()).dtype)
    bias = head["bias"].to(device=device, dtype=next(model.parameters()).dtype)
    model.eval()
    encoded = encode(model, weight, bias, tensors, args.batch_size, device)
    del model, raw_package, official_package, tensors
    gc.collect()

    records = []
    for vector, item in zip(encoded, metadata):
        q = embeddings[embedding_index[item["query_row"]]]
        p = embeddings[embedding_index[item["positive_row"]]]
        n = embeddings[embedding_index[item["negative_row"]]]
        baseline_margin = float(q @ p - q @ n)
        perturbed_margin = float(vector @ p - vector @ n)
        records.append(item | {
            "baseline_positive_similarity": float(q @ p),
            "baseline_negative_similarity": float(q @ n),
            "baseline_margin": baseline_margin,
            "perturbed_margin": perturbed_margin,
            "margin_change": perturbed_margin - baseline_margin,
            "embedding_preservation": float(vector @ q),
        })
    results = pd.DataFrame(records)
    key = [
        "query_row", "positive_row", "negative_row", "ik14", "formula", "adduct",
        "cross_condition_positive", "removed_count", "valid_peak_count",
    ]
    targeted = results.loc[results["condition"] == "targeted"].copy()
    controls = results.loc[results["condition"] == "matched_random"].groupby(key, as_index=False).agg(
        random_margin_change=("margin_change", "mean"),
        random_embedding_preservation=("embedding_preservation", "mean"),
        random_repeats_observed=("repeat", "count"),
    )
    paired = targeted.merge(controls, on=key, validate="one_to_one")
    paired = paired.loc[paired["random_repeats_observed"] == args.random_repeats].copy()
    if paired.empty:
        raise RuntimeError("no intervention retained all matched-random controls")
    paired["target_margin_change"] = paired["margin_change"]
    paired["target_minus_random_margin_change"] = paired["target_margin_change"] - paired["random_margin_change"]
    paired["baseline_status"] = np.where(paired["baseline_margin"] > 0, "baseline_correct", "baseline_wrong")

    groups = {
        "overall": paired,
        "cross_condition": paired.loc[paired["cross_condition_positive"]],
        "baseline_correct": paired.loc[paired["baseline_status"] == "baseline_correct"],
        "baseline_wrong": paired.loc[paired["baseline_status"] == "baseline_wrong"],
    }
    summaries = {
        name: summarize(group, args.bootstrap, args.seed + position)
        for position, (name, group) in enumerate(groups.items()) if len(group)
    }
    overall = summaries["overall"]
    cross = summaries.get("cross_condition")
    safe = summaries.get("baseline_correct")
    gates = {
        "identities_ge_300": overall["identities"] >= 300,
        "overall_identity_ci_positive": overall["identity_cluster_specificity_95ci"][0] > 0,
        "overall_formula_ci_positive": overall["formula_cluster_specificity_95ci"][0] > 0,
        "cross_condition_identity_ci_positive": bool(cross and cross["identity_cluster_specificity_95ci"][0] > 0),
        "cross_condition_formula_ci_positive": bool(cross and cross["formula_cluster_specificity_95ci"][0] > 0),
        "baseline_correct_absolute_identity_noninferior": bool(safe and safe["identity_cluster_absolute_change_95ci"][0] >= -0.002),
        "baseline_correct_absolute_formula_noninferior": bool(safe and safe["formula_cluster_absolute_change_95ci"][0] >= -0.002),
    }
    report = {
        "status": "directional_noise_v2_m1_complete",
        "selected_queries_before_token_filter": int(len(selected)),
        "paired_queries": int(len(paired)),
        "random_repeats": int(args.random_repeats),
        "results": summaries,
        "gates": gates,
        "pass": bool(all(gates.values())),
        "decision": "Only pass permits construction of a directional-noise fine-tuning pool; failure returns peak selection to M0 without training.",
        "provenance": {
            "m0_consensus_sha256": sha256(consensus_path),
            "p2_allow_sha256": sha256(allow_path),
            "embedding_cache_sha256": sha256(args.embedding_cache),
            "official_checkpoint_sha256": sha256(args.official_checkpoint),
        },
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.building-", dir=args.output_dir.parent))
    try:
        selected.to_csv(staging / "selected_triples.csv.gz", index=False, compression="gzip")
        results.to_csv(staging / "variant_results.csv.gz", index=False, compression="gzip")
        paired.to_csv(staging / "paired_margin_effects.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
