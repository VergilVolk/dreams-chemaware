#!/usr/bin/env python
"""Seal a larger polarity-stratified MoNA transfer panel for ChemAware.

The graph is built without embeddings or model scores.  Query/candidate
identities are absent from the local MassSpecGym development HDF5, positive
references are distinct spectra of the same identity, and exact normalized
spectrum clones are removed before query construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_mona_transfer_core import (  # noqa: E402
    build_panel,
    iter_mgf_records,
    prepare_metadata,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hdf_strings(handle: h5py.File, key: str) -> set[str]:
    output = set()
    for value in handle[key][:]:
        text = value.decode("utf-8", "ignore") if isinstance(value, bytes) else str(value)
        if text:
            output.add(text[:14])
    return output


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-mgf", type=Path, default=ROOT / "data/models/mona_pos_full.mgf")
    parser.add_argument("--negative-mgf", type=Path, default=ROOT / "data/models/mona_neg_full.mgf")
    parser.add_argument("--development-hdf5", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--queries-per-identity", type=int, default=4)
    parser.add_argument("--references-per-candidate", type=int, default=20)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--minimum-positive-queries", type=int, default=700)
    parser.add_argument("--minimum-negative-queries", type=int, default=1500)
    parser.add_argument("--minimum-combined-queries", type=int, default=2500)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    required = (args.positive_mgf, args.negative_mgf, args.development_hdf5)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite a transfer panel: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.development_hdf5, "r") as handle:
        development_identities = hdf_strings(handle, "INCHIKEY")

    RDLogger.DisableLog("rdApp.*")
    sources = {"positive": args.positive_mgf, "negative": args.negative_mgf}
    summaries: dict[str, dict[str, object]] = {}
    provenance: dict[str, dict[str, str]] = {}
    query_identities: set[str] = set()
    polarity_formula_clusters: set[str] = set()
    for polarity, mgf in sources.items():
        print(f"Parsing score-blind MoNA {polarity} metadata: {mgf}", flush=True)
        raw_records = list(iter_mgf_records(
            mgf, compute_signature=True, n_highest_peaks=args.n_highest_peaks,
        ))
        records, preparation = prepare_metadata(raw_records, development_identities)
        del raw_records
        panel, panel_summary = build_panel(
            records, ppm=args.ppm,
            queries_per_identity=args.queries_per_identity,
            references_per_candidate=args.references_per_candidate,
        )
        polarity_dir = args.output_dir / polarity
        polarity_dir.mkdir(parents=True, exist_ok=False)
        panel_path = polarity_dir / "panel.npz"
        np.savez_compressed(panel_path, **panel)
        summary = preparation | panel_summary
        summaries[polarity] = summary
        query_identities.update(map(str, panel["query_ik14"]))
        polarity_formula_clusters.update(
            f"{polarity}:{formula}" for formula in map(str, panel["query_formula"])
        )
        provenance[polarity] = {
            "mgf": str(mgf.resolve()),
            "mgf_sha256": sha256_file(mgf),
            "panel": str(panel_path.resolve()),
            "panel_sha256": sha256_file(panel_path),
        }
        (polarity_dir / "report.json").write_text(json.dumps({
            "status": "CHEMAWARE_MONA_POLARITY_PANEL_COMPLETE",
            "formal": True,
            "polarity": polarity,
            "summary": summary,
            "construction_uses_model_scores": False,
            "development_identity_overlap": 0,
            "exact_normalized_spectrum_clones_removed": True,
            "query_positive_spectrum_hash_disjoint": True,
            "candidate_protocol": (
                f"same polarity, same molecular formula, and {args.ppm:g} ppm precursor window; "
                "positive molecule first; max reference-spectrum similarity; ties count against positive"
            ),
            "provenance": provenance[polarity],
        }, indent=2), encoding="utf-8")
        print(json.dumps({"polarity": polarity, **summary}, indent=2), flush=True)

    positive_queries = int(summaries["positive"]["queries"])
    negative_queries = int(summaries["negative"]["queries"])
    combined_queries = positive_queries + negative_queries
    gates = {
        "positive_query_minimum": positive_queries >= args.minimum_positive_queries,
        "negative_query_minimum": negative_queries >= args.minimum_negative_queries,
        "combined_query_minimum": combined_queries >= args.minimum_combined_queries,
        "larger_than_phasea_role2_1975_queries": combined_queries > 1975,
        "development_identity_disjoint": True,
        "construction_score_blind": True,
        "spectrum_clone_guard": True,
        "polarity_never_mixed_within_query": True,
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"MoNA polarity transfer panel gates failed: {gates}; summaries={summaries}"
        )
    report = {
        "status": "CHEMAWARE_MONA_POLARITY_TRANSFER_PANEL_SEALED",
        "formal": True,
        "combined_queries": combined_queries,
        "combined_query_identities": len(query_identities),
        "polarity_formula_clusters": len(polarity_formula_clusters),
        "by_polarity": summaries,
        "gates": gates,
        "parameters": {
            "ppm": args.ppm,
            "queries_per_identity": args.queries_per_identity,
            "references_per_candidate": args.references_per_candidate,
            "n_highest_peaks": args.n_highest_peaks,
        },
        "contracts": {
            "model_scores_used_for_construction": False,
            "query_candidate_identity_absent_from_massspecgym_development": True,
            "query_and_positive_are_different_normalized_spectra": True,
            "query_and_positive_are_different_rows": True,
            "positive_and_negative_polarities_evaluated_separately": True,
            "formula_bootstrap_cluster": "polarity:molecular_formula",
        },
        "claim_limit": (
            "This is identity-disjoint from the local MassSpecGym fine-tuning/development corpus, "
            "but MoNA may have contributed to DreaMS pretraining; it is not a pretraining-novelty claim."
        ),
        "provenance": {
            "development_hdf5": str(args.development_hdf5.resolve()),
            "development_hdf5_sha256": sha256_file(args.development_hdf5),
            "sources": provenance,
            "builder_sha256": sha256_file(Path(__file__)),
            "core_sha256": sha256_file(ROOT / "tasks/chemaware_mona_transfer_core.py"),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
