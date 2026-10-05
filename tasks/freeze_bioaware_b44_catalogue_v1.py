#!/usr/bin/env python
"""Freeze the B42 KEGG+Rhea catalogue expert before B44 evaluation.

The model is trained only on the six opened B42 development domains.  Its
deployment gate is selected once from B42 cross-domain OOF predictions.  The
sealed B44 panel is not read here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b11_catalog_interaction_action import atomic_json, sha256  # noqa: E402
from audit_bioaware_b12_multicohort_catalog_action import build_universe  # noqa: E402
from audit_bioaware_b16_pairwise_nonlinear_action import fit_model  # noqa: E402
from audit_bioaware_b36_reaction_specificity_ablation import choose_gate, prepare_candidates  # noqa: E402
from audit_bioaware_b41_cross_catalog_topology import topology_signature  # noqa: E402
from audit_bioaware_b42_independent_catalog_topology import (  # noqa: E402
    CONSENSUS_FEATURES,
    add_independent_features,
    rhea_signature,
)
from evaluate_bioaware_b39_m2_fixed_action import atomic_csv_gzip  # noqa: E402


FEATURES = ["spectral_score", *CONSENSUS_FEATURES]


def atomic_joblib(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def catalogue_lookup(kegg_edges: Path, rhea_participants: Path) -> tuple[pd.DataFrame, dict]:
    kegg_members, kegg_degree, kegg_report = topology_signature(kegg_edges)
    rhea_members, rhea_degree, _, rhea_report = rhea_signature(rhea_participants)
    identities = sorted(kegg_members | rhea_members)
    frame = pd.DataFrame({"candidate_id": identities})
    frame["strict_kegg_member"] = frame["candidate_id"].isin(kegg_members).astype(float)
    frame["strict_kegg_log_degree"] = frame["candidate_id"].map(kegg_degree).fillna(0.0)
    frame["rhea_member"] = frame["candidate_id"].isin(rhea_members).astype(float)
    frame["rhea_log_degree"] = frame["candidate_id"].map(rhea_degree).fillna(0.0)
    frame["independent_member_count"] = frame["strict_kegg_member"] + frame["rhea_member"]
    frame["independent_member_intersection"] = frame["strict_kegg_member"] * frame["rhea_member"]
    frame["independent_log_degree_mean"] = 0.5 * (
        frame["strict_kegg_log_degree"] + frame["rhea_log_degree"]
    )
    frame["independent_log_degree_min"] = np.minimum(
        frame["strict_kegg_log_degree"], frame["rhea_log_degree"]
    )
    return frame, {"strict_kegg": kegg_report, "rhea": rhea_report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-candidates", type=Path, default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz")
    parser.add_argument("--st-candidates", type=Path, default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz")
    parser.add_argument("--st-queries", type=Path, default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz")
    parser.add_argument("--kgmn-candidates", type=Path, default=ROOT / "data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz")
    parser.add_argument("--kgmn-seeds", type=Path, default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz")
    parser.add_argument("--kegg-edges", type=Path, default=ROOT / "data/reference/metdna2_kegg_network_20260828/metdna2_kegg_edges.csv.gz")
    parser.add_argument("--emrn-edges", type=Path, default=ROOT / "data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz")
    parser.add_argument("--rhea-participants", type=Path, default=ROOT / "data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz")
    parser.add_argument("--b42-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    required = [
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds, args.kegg_edges, args.emrn_edges,
        args.rhea_participants, args.b42_dir / "report.json",
        args.b42_dir / "candidate_catalog_features.csv.gz",
        args.b42_dir / "cross_catalog_transitions.csv.gz",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    b42_report = json.loads((args.b42_dir / "report.json").read_text(encoding="utf-8"))
    if b42_report.get("status") != "bioaware_b42_independent_catalog_topology_complete":
        raise RuntimeError("B42 report status is not the validated independent-catalogue audit")

    universe, universe_report = build_universe(args)
    candidates = prepare_candidates(universe)
    candidates, _ = add_independent_features(
        candidates, args.kegg_edges, args.emrn_edges, args.rhea_participants,
        null_repeats=3, seed=args.seed,
    )
    frozen_features = pd.read_csv(args.b42_dir / "candidate_catalog_features.csv.gz")
    columns = ["query_id", "candidate_id", *CONSENSUS_FEATURES]
    joined = candidates[columns].merge(
        frozen_features[columns], on=["query_id", "candidate_id"], how="outer",
        validate="one_to_one", suffixes=("_rebuilt", "_b42"), indicator=True,
    )
    maximum_error = max(
        float(np.max(np.abs(
            joined[f"{column}_rebuilt"].to_numpy(float)
            - joined[f"{column}_b42"].to_numpy(float)
        ), initial=0.0))
        for column in CONSENSUS_FEATURES
    )
    feature_replay = {
        "rows": int(len(joined)),
        "key_mismatches": int((joined["_merge"] != "both").sum()),
        "maximum_consensus_feature_error": maximum_error,
    }
    if feature_replay["rows"] != 6695 or feature_replay["key_mismatches"] or maximum_error > 1e-12:
        raise RuntimeError(f"B42 consensus feature replay failed: {feature_replay}")

    transitions = pd.read_csv(args.b42_dir / "cross_catalog_transitions.csv.gz")
    oof = transitions.loc[transitions["arm"].eq("kegg_rhea_consensus")].copy()
    if len(oof) != 860 or oof["query_id"].nunique() != 860:
        raise RuntimeError("B42 consensus OOF transition coverage drift")
    selected_gate, gate_ledger = choose_gate(oof)
    if selected_gate["gate_name"] == "no_op" or selected_gate["risk_net_lambda2"] <= 0:
        raise RuntimeError(f"B42 OOF did not select a useful global deployment gate: {selected_gate}")

    model, fit_report = fit_model(candidates, FEATURES, args.seed + 44000)
    lookup, catalogue_report = catalogue_lookup(args.kegg_edges, args.rhea_participants)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.joblib"
    lookup_path = args.output_dir / "catalogue_lookup.csv.gz"
    ledger_path = args.output_dir / "oof_gate_ledger.csv.gz"
    atomic_joblib(model_path, model)
    atomic_csv_gzip(lookup_path, lookup)
    atomic_csv_gzip(ledger_path, pd.DataFrame(gate_ledger))
    report = {
        "status": "bioaware_catalogue_v1_frozen",
        "formal": True,
        "feature_family": "strict_KEGG_plus_currency_filtered_Rhea_candidate_catalogue_topology",
        "features": FEATURES,
        "model_class": type(model).__name__,
        "model_parameters": model.get_params(deep=False),
        "fit": fit_report,
        "deployment_gate": selected_gate,
        "gate_source": "single selection on six-domain B42 OOF predictions; B44 unread",
        "feature_replay": feature_replay,
        "catalogues": catalogue_report,
        "contracts": {
            "candidate_static_prior": True,
            "sample_context_used": False,
            "phenotype_used": False,
            "P2b_used": False,
            "B44_read": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            **universe_report["provenance"],
            "b42_report": sha256(args.b42_dir / "report.json"),
            "b42_candidate_features": sha256(args.b42_dir / "candidate_catalog_features.csv.gz"),
            "b42_transitions": sha256(args.b42_dir / "cross_catalog_transitions.csv.gz"),
            "kegg_edges": sha256(args.kegg_edges),
            "rhea_participants": sha256(args.rhea_participants),
            "model": sha256(model_path),
            "catalogue_lookup": sha256(lookup_path),
            "gate_ledger": sha256(ledger_path),
            "script": sha256(Path(__file__)),
            "scikit_learn_version": sklearn.__version__,
        },
        "claim_limit": "Frozen post-embedding candidate expert. B42 is opened development evidence; external performance is unknown until one-time B44 evaluation.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
