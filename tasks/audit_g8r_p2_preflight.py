"""Fail-closed preflight for P2 listwise development.

This script does not train a model.  It proves three prerequisites:

1. the sealed P3 identities are excluded from every P2 training row;
2. the real P2 pool actually contains deployable strict-10ppm candidate lists;
3. the frozen RAW-v1 artifact exactly reproduces the archived +4.35 pp result
   on the already-consumed compatibility set.

P3 scores are never read or computed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_PAIRS = ROOT / "tasks/massspecgym_isomers/pairs.json"
DEFAULT_P3 = ROOT / "data/validation/g8r_p3_test"
DEFAULT_ARTIFACT = ROOT / "data/validation/g8r_raw_reranker_v1_artifact.json"
DEFAULT_TRAIN_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache.npz"
DEFAULT_VAL_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache_val.npz"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_p2_preflight.json"

EXPECTED_HDF5_SHA256 = "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f"
EXPECTED_PAIRS_SHA256 = "0461e95b7e98d9d7fa0f03c884b7b8b7996e58bbdafecdb95de31e02c6cb0d9a"
EXPECTED_LIBRARY_SHA256 = "cad6ee05aee1ccfb8d4923b2405795a5538cb1876247b0d098db033343361385"
EXPECTED_BASE_R1 = 0.8080645161290323
EXPECTED_RAW_R1 = 0.8516129032258064
EXPECTED_CORRECTED = 44
EXPECTED_INTRODUCED = 17
EXPECTED_COVERAGE = 0.4645161290322581


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--p3-dir", type=Path, default=DEFAULT_P3)
    p.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    p.add_argument("--train-cache", type=Path, default=DEFAULT_TRAIN_CACHE)
    p.add_argument("--val-cache", type=Path, default=DEFAULT_VAL_CACHE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--ppm-tol", type=float, default=10.0)
    p.add_argument("--min-real-identities", type=int, default=3000)
    p.add_argument("--min-valid-identities", type=int, default=1500)
    p.add_argument("--min-near-identities", type=int, default=150)
    return p.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_text_array(dataset) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8", "ignore") if isinstance(value, bytes) else str(value)
        for value in dataset[:]
    ], dtype=object)


def read_cache(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as cache:
        return pd.DataFrame({name: cache[name] for name in cache.files})


def strict_retrieval(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Exact archived RAW-v1 molecule-ranking convention."""
    records = []
    for query, group in frame.groupby("query", sort=False):
        positives = group[group["label"] == 1]
        negatives = group[group["label"] == 0]
        if positives.empty or negatives.empty:
            continue
        positive_score = float(positives[score_col].max())
        molecule_scores = group.groupby("candidate_ik14", sort=False)[score_col].max()
        negative_scores = negatives.groupby("candidate_ik14", sort=False)[score_col].max().to_numpy(float)
        ordered = molecule_scores.sort_values(ascending=False, kind="mergesort")
        confidence = float(ordered.iloc[0] - ordered.iloc[1]) if len(ordered) > 1 else float("inf")
        rank = 1 + int(np.sum(negative_scores >= positive_score))
        records.append({
            "query": int(query),
            "formula": str(group.iloc[0]["formula"]),
            "top1": rank == 1,
            "rank": rank,
            "mrr": 1.0 / rank,
            "confidence": confidence,
        })
    return pd.DataFrame(records).sort_values("query").reset_index(drop=True)


def frozen_raw_reproduction(artifact_path: Path, train_cache: Path, val_cache: Path) -> dict:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("format") != "raw_reranker_v1_artifact":
        raise RuntimeError("unexpected RAW-v1 artifact format")
    if sha256_file(train_cache) != artifact.get("train_cache_sha256"):
        raise RuntimeError("RAW-v1 artifact does not match the frozen training cache")

    frame = read_cache(val_cache)
    features = list(artifact["feature_names"])
    missing = sorted(set(features + ["query", "label", "candidate_ik14", "formula"]) - set(frame.columns))
    if missing:
        raise RuntimeError(f"validation cache is missing fields: {missing}")
    mean = np.asarray(artifact["scaler_mean"], dtype=np.float64)
    scale = np.asarray(artifact["scaler_scale"], dtype=np.float64)
    coef = np.asarray(artifact["model_coef"], dtype=np.float64)
    if len(features) != len(mean) or mean.shape != scale.shape or mean.shape != coef.shape:
        raise RuntimeError("frozen RAW-v1 artifact has inconsistent dimensions")
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError("frozen RAW-v1 scaler is invalid")
    x = frame[features].to_numpy(np.float64)
    if np.any(~np.isfinite(x)):
        raise RuntimeError("non-finite feature in compatibility cache")
    frame["rawv1_score"] = ((x - mean) / scale) @ coef + float(artifact.get("model_intercept", 0.0))

    baseline = strict_retrieval(frame, "dreams_similarity")
    reranked = strict_retrieval(frame, "rawv1_score")
    if not np.array_equal(baseline["query"].to_numpy(), reranked["query"].to_numpy()):
        raise RuntimeError("baseline/reranker query mismatch")
    gate = baseline["confidence"].to_numpy() <= float(artifact["gate_threshold"])
    base_top1 = baseline["top1"].to_numpy(bool)
    raw_top1 = reranked["top1"].to_numpy(bool)
    final_top1 = np.where(gate, raw_top1, base_top1)
    result = {
        "n_queries": int(len(baseline)),
        "baseline_recall1": float(base_top1.mean()),
        "raw_v1_gated_recall1": float(final_top1.mean()),
        "delta_recall1": float(final_top1.mean() - base_top1.mean()),
        "corrected": int(np.sum((~base_top1) & final_top1)),
        "introduced": int(np.sum(base_top1 & (~final_top1))),
        "gate_coverage": float(gate.mean()),
        "artifact_sha256": sha256_file(artifact_path),
        "train_cache_sha256": sha256_file(train_cache),
        "val_cache_sha256": sha256_file(val_cache),
    }
    checks = [
        abs(result["baseline_recall1"] - EXPECTED_BASE_R1) < 1e-12,
        abs(result["raw_v1_gated_recall1"] - EXPECTED_RAW_R1) < 1e-12,
        result["corrected"] == EXPECTED_CORRECTED,
        result["introduced"] == EXPECTED_INTRODUCED,
        abs(result["gate_coverage"] - EXPECTED_COVERAGE) < 1e-12,
    ]
    if not all(checks):
        raise RuntimeError(f"archived +4.35 pp compatibility gate failed: {result}")
    result["pass"] = True
    return result


def p3_manifest_ik14(p3_dir: Path) -> set[str]:
    identities: set[str] = set()
    manifests = sorted(p3_dir.glob("p3_*_manifest.json"))
    if len(manifests) != 6:
        raise RuntimeError(f"expected six sealed P3 manifests, found {len(manifests)}")
    for path in manifests:
        body = json.loads(path.read_text(encoding="utf-8"))
        identities.update(str(query["ik14"]) for query in body["queries"])
    return identities


def near_graph(path: Path) -> dict[str, set[str]]:
    body = json.loads(path.read_text(encoding="utf-8"))
    graph: dict[str, set[str]] = defaultdict(set)
    for row in body.get("near", []):
        a, b = str(row["ik_a"]), str(row["ik_b"])
        graph[a].add(b)
        graph[b].add(a)
    return graph


def audit_training_pool(a: argparse.Namespace, p3_ik14: set[str]) -> dict:
    allow_path = a.p3_dir / "p3_p2_allowed_training_ik14.json"
    allow = json.loads(allow_path.read_text(encoding="utf-8"))
    primary = allow["real_train_primary"]
    rows = np.asarray(primary["rows"], dtype=np.int64)
    allowed_ik = {str(value) for value in primary["ik14"]}
    if primary["n"] != len(allowed_ik) or primary["n_rows"] != len(rows):
        raise RuntimeError("P2 allow-list counts do not match their bodies")
    if allowed_ik & p3_ik14:
        raise RuntimeError("sealed P3 identity leaked into P2 training identities")
    if len(allowed_ik) < a.min_real_identities:
        raise RuntimeError(f"too few real P2 identities: {len(allowed_ik)}")

    with h5py.File(a.data, "r") as h:
        n_rows = len(h["precursor_mz"])
        if len(rows) == 0 or rows.min() < 0 or rows.max() >= n_rows or len(np.unique(rows)) != len(rows):
            raise RuntimeError("P2 real row allow-list is invalid")
        pmz = np.asarray(h["precursor_mz"][rows], dtype=np.float64)
        ik14 = np.asarray([value[:14] for value in read_text_array(h["INCHIKEY"])[rows]], dtype=object)
        adduct = read_text_array(h["adduct"])[rows]
        fold = read_text_array(h["fold"])[rows]
        simulation = read_text_array(h["SIMULATION_CHALLENGE"])[rows]
        formula = read_text_array(h["FORMULA"])[rows]
    if set(ik14) != allowed_ik:
        raise RuntimeError("row-derived identities do not equal the P2 identity allow-list")
    if np.any(fold != "train"):
        raise RuntimeError("non-train fold row leaked into P2 real training pool")
    if np.any(simulation != "False"):
        raise RuntimeError("simulated row leaked into P2 real training pool")
    if np.any(~np.isfinite(pmz)) or np.any(pmz <= 0):
        raise RuntimeError("invalid precursor mass in P2 real training pool")

    order = np.argsort(pmz, kind="mergesort")
    sorted_mass = pmz[order]
    near = near_graph(a.pairs)
    valid_ik: set[str] = set()
    near_valid_ik: set[str] = set()
    valid_queries = 0
    for local_row in range(len(rows)):
        mass = pmz[local_row]
        delta = a.ppm_tol * 1e-6 * mass
        lo = np.searchsorted(sorted_mass, mass - delta, side="left")
        hi = np.searchsorted(sorted_mass, mass + delta, side="right")
        candidate = order[lo:hi]
        candidate = candidate[(candidate != local_row) & (adduct[candidate] == adduct[local_row])]
        if len(candidate) == 0:
            continue
        qik = str(ik14[local_row])
        candidate_ik = {str(ik14[index]) for index in candidate}
        if qik not in candidate_ik or not (candidate_ik - {qik}):
            continue
        valid_queries += 1
        valid_ik.add(qik)
        if near.get(qik, set()) & candidate_ik:
            near_valid_ik.add(qik)

    result = {
        "n_rows": int(len(rows)),
        "n_identities": int(len(allowed_ik)),
        "n_formulas": int(len(set(formula))),
        "n_valid_query_rows": int(valid_queries),
        "n_valid_identities": int(len(valid_ik)),
        "n_valid_near_identities": int(len(near_valid_ik)),
        "p3_identity_overlap": 0,
        "all_real": True,
        "all_train_fold": True,
    }
    if result["n_valid_identities"] < a.min_valid_identities:
        raise RuntimeError(f"insufficient deployable P2 identity lists: {result}")
    if result["n_valid_near_identities"] < a.min_near_identities:
        raise RuntimeError(f"insufficient strict-10ppm near identities: {result}")
    result["pass"] = True
    return result


def main() -> None:
    a = parse_args()
    required = [a.data, a.pairs, a.artifact, a.train_cache, a.val_cache,
                a.p3_dir / "p3_lock_summary.json",
                a.p3_dir / "p3_p2_allowed_training_ik14.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P2 preflight inputs missing: {missing}")

    summary = json.loads((a.p3_dir / "p3_lock_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "g8r_p3_v3_sealed":
        raise RuntimeError("P3 is not formally sealed")
    hashes = {
        "hdf5": sha256_file(a.data),
        "pairs": sha256_file(a.pairs),
        "reference_library": summary.get("reference_library_sha256"),
    }
    expected = {
        "hdf5": EXPECTED_HDF5_SHA256,
        "pairs": EXPECTED_PAIRS_SHA256,
        "reference_library": EXPECTED_LIBRARY_SHA256,
    }
    if hashes != expected:
        raise RuntimeError(f"sealed-input hash mismatch: actual={hashes}, expected={expected}")
    if summary.get("hdf5_sha256") != hashes["hdf5"] or summary.get("pairs_json_sha256") != hashes["pairs"]:
        raise RuntimeError("P3 summary does not match current HDF5/pairs files")

    heldout = p3_manifest_ik14(a.p3_dir)
    pool = audit_training_pool(a, heldout)
    compatibility = frozen_raw_reproduction(a.artifact, a.train_cache, a.val_cache)
    report = {
        "status": "g8r_p2_preflight_passed",
        "sealed_hashes": hashes,
        "n_p3_query_identities": len(heldout),
        "training_pool": pool,
        "raw_v1_compatibility_reproduction": compatibility,
        "interpretation": (
            "The archived +4.35 pp compatibility result is reproduced on the consumed "
            "g8r_val protocol. This is an implementation-identity gate, not a promise "
            "of +4.35 pp on sealed P3."
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix(a.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(a.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[P2 preflight] PASS: {a.output}")


if __name__ == "__main__":
    main()
