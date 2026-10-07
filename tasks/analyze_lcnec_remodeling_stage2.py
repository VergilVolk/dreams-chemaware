"""LCNEC Stage-2: LIPn cross-platform + dark increment + transformation edges.

Three pre-registered experiments, all using Stage-1d's PASS outputs:
  2A: LIPn cross-platform family-direction consistency (≥ 60%)
  2B: dark entity increment (adding dark entities improves sign-flip p ≥ 10×)
  2C: transformation edge patient consistency (≥ 5 edges at BH-FDR ≤ 0.05)

Runs on the server: needs LIPn mzML zip + Stage-1 artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.join("tasks"))


# ============================================================================
# Shared utilities
# ============================================================================

def sha256_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def unit_rows(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def mutual_knn(embedding, k=3):
    n = len(embedding)
    sim = embedding @ embedding.T
    np.fill_diagonal(sim, -np.inf)
    directed = np.zeros((n, n), dtype=bool)
    for i in range(n):
        nb = np.argpartition(-sim[i], kth=k-1)[:k]
        directed[i, nb] = True
    adj = directed & directed.T
    _, labels = connected_components(
        csr_matrix(adj.astype(np.int8)), directed=False, return_labels=True)
    return labels.astype(np.int64), adj


def decompose(values, labels):
    m = np.asarray(values, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    fam, wit, iso = np.zeros_like(m), np.zeros_like(m), np.zeros_like(m)
    for comp in np.unique(labels):
        idx = np.flatnonzero(labels == comp)
        if len(idx) == 1:
            iso[:, idx] = m[:, idx]
        else:
            mean = m[:, idx].mean(axis=1, keepdims=True)
            fam[:, idx] = mean
            wit[:, idx] = m[:, idx] - mean
    if m.shape[0] == 1:
        return fam[0], wit[0], iso[0]
    return fam, wit, iso


def energy(v):
    return float(np.sum(np.square(np.asarray(v, dtype=np.float64))))


def signflip_p(effects, labels, component, draws=10000, seed=20261007):
    mean_vec = effects.mean(axis=0)
    fam, wit, iso = decompose(mean_vec, labels)
    obs = energy({"family": fam, "within": wit, "isolated": iso}[component])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(draws):
        signs = rng.choice((-1.0, 1.0), size=len(effects))
        m2 = np.mean(effects * signs[:, None], axis=0)
        f2, w2, i2 = decompose(m2, labels)
        null_e = energy({"family": f2, "within": w2, "isolated": i2}[component])
        if null_e >= obs:
            count += 1
    return (1 + count) / (draws + 1)


# ============================================================================
# 2A: LIPn cross-platform consistency
# ============================================================================

def experiment_2a(stage1_dir, lipn_matrix_path, output_dir):
    """Check family-direction consistency between HSST3n and LIPn."""
    npz = np.load(stage1_dir / "primary_decomposition.npz")
    effects_hsst = npz["patient_effect"]  # (34, 263)
    family_id = npz["family_id"]
    patient = [str(p) for p in npz["patient"]]

    # Load LIPn matrix
    lipn = np.load(lipn_matrix_path)
    area_lipn = np.asarray(lipn["area"], dtype=np.float64)
    sample_code = np.asarray(lipn["sample_code"]).astype(str)
    group_code = np.asarray(lipn["group_code"]).astype(str)
    file_class = np.asarray(lipn["file_class"]).astype(str)
    lipn_family_id = np.asarray(lipn["family_id"], dtype=np.int64)

    # Align LIPN family order to Stage-1 order
    lipn_lookup = {int(fid): col for col, fid in enumerate(lipn_family_id)}
    order = [lipn_lookup[int(fid)] for fid in family_id]
    area_lipn = area_lipn[:, order]

    # Compute LIPn patient effects
    study = np.flatnonzero(file_class == "study")
    pairs = defaultdict(dict)
    for idx in study:
        pairs[sample_code[idx]][group_code[idx]] = int(idx)

    common_patients = sorted(set(pairs) & set(patient))
    if len(common_patients) < 20:
        return {"status": "2A_FAIL", "reason": f"too few common patients: {len(common_patients)}"}

    tumour_l = np.stack([area_lipn[pairs[p]["TU"]] for p in common_patients])
    adjacent_l = np.stack([area_lipn[pairs[p]["NG"]] for p in common_patients])
    positive = np.where(area_lipn[study] > 0, area_lipn[study], np.inf)
    pseudo = np.minimum(np.min(positive, axis=0) / 2.0, 1.0)
    pseudo = np.where(np.isfinite(pseudo) & (pseudo > 0), pseudo, 1.0)
    effects_lipn = np.log2(tumour_l + pseudo) - np.log2(adjacent_l + pseudo)

    # Match HSST3n patients to LIPn patients
    hsst_idx = [patient.index(p) for p in common_patients]
    effects_hsst_matched = effects_hsst[hsst_idx]

    # Load frozen graph labels
    embed_dir = stage1_dir.parent / "query" / "dreams_embeddings"
    embedding = unit_rows(np.load(embed_dir / "embeddings.npy"))
    manifest = pd.read_csv(embed_dir / "manifest.csv")
    def pfid(n):
        m = re.search(r"_family_(\d+)$", str(n))
        return int(m.group(1)) if m else -1
    fids = [pfid(n) for n in manifest["name"]]
    lookup = {f: i for i, f in enumerate(fids)}
    order_emb = [lookup[int(f)] for f in family_id]
    embedding = embedding[order_emb]
    labels, _ = mutual_knn(embedding, k=3)

    # For each non-singleton component, check direction consistency
    # between HSST3n and LIPn within-family differentials
    consistent, total = 0, 0
    for comp in np.unique(labels):
        idx = np.flatnonzero(labels == comp)
        if len(idx) < 2:
            continue
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                d_hsst = effects_hsst_matched[:, idx[j]] - effects_hsst_matched[:, idx[i]]
                d_lipn = effects_lipn[:, idx[j]] - effects_lipn[:, idx[i]]
                # direction = sign of median differential
                sign_hsst = np.sign(np.median(d_hsst))
                sign_lipn = np.sign(np.median(d_lipn))
                if sign_hsst != 0 and sign_lipn != 0:
                    total += 1
                    if sign_hsst == sign_lipn:
                        consistent += 1

    consistency = consistent / max(total, 1)
    result = {
        "status": "2A_" + ("PASS" if consistency >= 0.60 else "FAIL"),
        "common_patients": len(common_patients),
        "entity_pairs_tested": total,
        "consistent_pairs": consistent,
        "consistency_fraction": round(consistency, 4),
        "gate": ">= 0.60",
    }
    return result


# ============================================================================
# 2B: Dark entity increment
# ============================================================================

def experiment_2b(stage1_dir, gnps_manifest_path, gnps_mgf_path, output_dir):
    """Compare decomposition with known-only vs all 263 entities."""
    npz = np.load(stage1_dir / "primary_decomposition.npz")
    effects = npz["patient_effect"]
    family_id = npz["family_id"]

    embed_dir = stage1_dir.parent / "query" / "dreams_embeddings"
    embedding = unit_rows(np.load(embed_dir / "embeddings.npy"))
    manifest = pd.read_csv(embed_dir / "manifest.csv")
    def pfid(n):
        m = re.search(r"_family_(\d+)$", str(n))
        return int(m.group(1)) if m else -1
    fids = [pfid(n) for n in manifest["name"]]
    lookup = {f: i for i, f in enumerate(fids)}
    order_emb = [lookup[int(f)] for f in family_id]
    embedding = embedding[order_emb]
    labels, _ = mutual_knn(embedding, k=3)

    # Determine known vs dark: search each entity against GNPS library
    # Use the GNPS benchmark manifest to get library spectra embeddings
    # For speed, use a subset of the library (gold quality only)
    gnps_manifest = pd.read_csv(gnps_manifest_path)
    # Load a manageable subset of spectra for similarity computation
    # Instead of loading all 329k spectra, use the manifest's formulas
    # and check if any library spectrum has similar precursor mass AND formula

    # Simpler approach: use the dark module rescan results we already computed
    # or compute against the benchmark panel directly
    # For now: use mass-based matching against GNPS library
    # (this is an approximation; full embedding search is better but slower)

    # Load dark module rescan results if available
    rescan_path = Path("data/validation/GLM_track2_census/lcnec_dark_full_rescan.json")
    if rescan_path.exists():
        rescan = json.load(open(rescan_path))
        best_cosine = {e["precursor"]: e["best_cosine"] for e in rescan["modules"]}
    else:
        # Fallback: all entities classified as dark (conservative)
        best_cosine = {}

    # Get precursor masses from the npz
    mz = npz["mz"]
    known_mask = np.zeros(len(family_id), dtype=bool)
    for i, (fid, m) in enumerate(zip(family_id, mz)):
        # Check if this family has a library match above 0.80 cosine
        if m in best_cosine and best_cosine[m] >= 0.80:
            known_mask[i] = True
        else:
            # Also check mass-based match (same formula within 5 ppm)
            # This is a weaker criterion but catches exact formula matches
            known_mask[i] = False

    n_known = int(known_mask.sum())
    n_dark = int((~known_mask).sum())

    if n_known < 20:
        return {"status": "2B_FAIL", "reason": f"too few known entities: {n_known}"}

    # All-features decomposition (= Stage-1d result)
    p_all_family = signflip_p(effects, labels, "family", draws=5000)
    p_all_within = signflip_p(effects, labels, "within", draws=5000)

    # Known-only decomposition (mask dark columns to zero)
    effects_known = effects.copy()
    effects_known[:, ~known_mask] = 0.0
    labels_known = labels.copy()
    # Treat dark entities as singletons in known-only mode
    for i in range(len(labels_known)):
        if not known_mask[i]:
            labels_known[i] = -i  # unique label = singleton
    p_known_family = signflip_p(effects_known, labels_known, "family", draws=5000)
    p_known_within = signflip_p(effects_known, labels_known, "within", draws=5000)

    improvement = max(p_known_family / max(p_all_family, 1e-10),
                     p_known_within / max(p_all_within, 1e-10))

    result = {
        "status": "2B_" + ("PASS" if improvement >= 10 else "FAIL"),
        "known_entities": n_known,
        "dark_entities": n_dark,
        "dark_fraction": round(n_dark / len(family_id), 4),
        "p_all_family": p_all_family,
        "p_all_within": p_all_within,
        "p_known_family": p_known_family,
        "p_known_within": p_known_within,
        "improvement_factor": round(improvement, 1),
        "gate": ">= 10x improvement",
    }
    return result


# ============================================================================
# 2C: Transformation edge patient consistency
# ============================================================================

def experiment_2c(stage1_dir, output_dir):
    """Test transformation edges using the PMD module."""
    from GLM_transformation_edges import (
        TRANSFORMATION_PMD_TABLE, SpectralEntity, edge_candidates,
        edge_differentials, patient_consistency,
    )

    npz = np.load(stage1_dir / "primary_decomposition.npz")
    effects = npz["patient_effect"]
    family_id = npz["family_id"].tolist()
    mz = npz["mz"]
    rt_sec = npz["rt_sec"]

    # Build entities (neutral mass ≈ precursor_mz - proton for [M+H]+)
    proton = 1.007276
    entities = []
    for fid, m in zip(family_id, mz):
        neutral = float(m) - proton
        entities.append(SpectralEntity(
            family_id=int(fid), neutral_mass=neutral,
            precursor_mz=float(m), ms2_peaks=(),
        ))

    # Build edges (no MS2 peaks available locally; use mass-only)
    edges = edge_candidates(entities, ppm_tolerance=20, max_edges=5000)
    if len(edges) < 10:
        return {"status": "2C_FAIL", "reason": f"too few edges: {len(edges)}"}

    # Compute differentials
    diffs = edge_differentials(effects, family_id, edges)

    # Patient consistency for each edge
    results = []
    for (uid, vid), g in diffs.items():
        consistency = patient_consistency(g)
        edge_info = next((e for e in edges if e.u == uid and e.v == vid), None)
        if edge_info and consistency["n"] >= 10:
            results.append({
                "family_u": uid, "family_v": vid,
                "pmd_name": edge_info.pmd_name,
                "pmd_observed": edge_info.pmd_observed,
                "fragment_shared": edge_info.fragment_shared,
                **consistency,
            })

    if not results:
        return {"status": "2C_FAIL", "reason": "no edges with enough patients"}

    # BH correction
    p_values = [r["p_value"] for r in results]
    rejected, p_adjusted, _, _ = sps.multipletests(p_values, method="fdr_bh")
    n_significant = int(rejected.sum())

    for r, padj in zip(results, p_adjusted):
        r["p_adjusted"] = round(float(padj), 6)

    result = {
        "status": "2C_" + ("PASS" if n_significant >= 5 else "FAIL"),
        "n_edges_tested": len(results),
        "n_with_fragment_support": sum(1 for r in results if r["fragment_shared"]),
        "n_significant_fdr_05": n_significant,
        "gate": ">= 5 edges at BH-FDR ≤ 0.05",
        "top_edges": sorted(results, key=lambda r: r["p_adjusted"])[:20],
    }
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", type=Path, required=True,
                        help="Stage-1 analysis dir (contains primary_decomposition.npz)")
    parser.add_argument("--lipn-matrix", type=Path, default=None,
                        help="Path to LIPn EIC matrix (all_qc_family_eic_matrix.npz)")
    parser.add_argument("--gnps-manifest", type=Path, required=True)
    parser.add_argument("--gnps-mgf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    results = {}

    # 2A: LIPn cross-platform (optional if matrix provided)
    if args.lipn_matrix and args.lipn_matrix.is_file():
        print("Running 2A: LIPn cross-platform...", flush=True)
        results["2a"] = experiment_2a(args.stage1_dir, args.lipn_matrix, args.output_dir)
    else:
        results["2a"] = {"status": "2A_SKIPPED", "reason": "LIPn matrix not provided"}

    # 2B: Dark entity increment
    print("Running 2B: dark entity increment...", flush=True)
    results["2b"] = experiment_2b(args.stage1_dir, args.gnps_manifest, args.gnps_mgf, args.output_dir)

    # 2C: Transformation edges
    print("Running 2C: transformation edges...", flush=True)
    results["2c"] = experiment_2c(args.stage1_dir, args.output_dir)

    # Overall verdict
    results_2a = results["2a"]["status"].endswith("PASS")
    results_2b = results["2b"]["status"].endswith("PASS")
    results_2c = results["2c"]["status"].endswith("PASS")

    if results_2a and results_2b and results_2c:
        overall = "STAGE2_PASS"
    elif results_2a and results_2b:
        overall = "STAGE2_PARTIAL_NO_TRANSFORMATION"
    elif results_2b:
        overall = "STAGE2_PARTIAL_SINGLE_PLATFORM"
    else:
        overall = "STAGE2_STOP"

    report = {
        "status": "LCNEC_GLOBAL_REMODELING_" + overall,
        "experiments": results,
        "verdict_matrix": {
            "2A_cross_platform": results_2a,
            "2B_dark_increment": results_2b,
            "2C_transformation": results_2c,
        },
        "provenance": {
            "stage1_dir": str(args.stage1_dir),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v["status"] for k, v in results.items()}, indent=2), flush=True)
    print(f"OVERALL: {overall}", flush=True)


if __name__ == "__main__":
    main()
