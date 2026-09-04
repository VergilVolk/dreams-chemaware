#!/usr/bin/env python
"""Prepare the fixed MTBLS1905 known-metabolite BioAware benchmark.

Two seed regimes are emitted:

``auto``
    Deployable, phenotype-blind high-confidence DreaMS Top-1 seeds.

``published_leave_target_out``
    Evaluation-only network headroom.  Other published identities are seeds;
    the evaluator removes the held-out truth identity for each query.  This
    regime cannot be reported as deployment performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ik14(value: object) -> str:
    return str(value).split("-", 1)[0]


def formula(smiles: object) -> str:
    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) and str(smiles) else None
    return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=Path("data/external/MTBLS1905/reference/blind_connectivity_panel.tsv"))
    parser.add_argument("--target-matches", type=Path, default=Path("data/external/MTBLS1905/qc_ms2/audit/published_target_qc_ms2_matches.tsv"))
    parser.add_argument("--candidate-map", type=Path, default=Path("data/external/MTBLS1905/reference/hnscc_positive_mass_candidate_map.tsv"))
    parser.add_argument("--query-dir", type=Path, default=Path("data/external/MTBLS1905/qc_ms2/dreams_official_full"))
    parser.add_argument("--reference-dir", type=Path, default=Path("data/external/MTBLS1905/reference/hnscc_positive_reference_dreams"))
    parser.add_argument("--participants", type=Path, default=Path("data/reference/bioaware_rhea/rhea_participants.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/MTBLS1905/bioaware_v1_input"))
    parser.add_argument("--auto-seed-minimum-score", type=float, default=0.80)
    parser.add_argument("--auto-seed-minimum-margin", type=float, default=0.05)
    parser.add_argument("--ppm-tolerance", type=float, default=10.0)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"fail-closed: output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    required_paths = [
        args.panel, args.target_matches, args.candidate_map, args.participants,
        args.query_dir / "manifest.csv", args.query_dir / "official_embeddings.npy",
        args.reference_dir / "manifest.csv", args.reference_dir / "embeddings.npy",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    panel = pd.read_csv(args.panel, sep="\t")
    panel = panel[panel.panel_status.eq("evaluable")].set_index("metabolite")
    matches = pd.read_csv(args.target_matches, sep="\t")
    matches = matches[matches.metabolite.isin(panel.index)].copy()
    candidate_map = pd.read_csv(args.candidate_map, sep="\t")
    query_manifest = pd.read_csv(args.query_dir / "manifest.csv")
    query_embeddings = np.load(args.query_dir / "official_embeddings.npy", mmap_mode="r")
    ref_manifest = pd.read_csv(args.reference_dir / "manifest.csv")
    ref_embeddings = np.load(args.reference_dir / "embeddings.npy", mmap_mode="r")
    if len(ref_manifest) != len(ref_embeddings) or len(query_manifest) != len(query_embeddings):
        raise RuntimeError("embedding/manifest rows mismatch")
    graph_compounds = set(pd.read_csv(args.participants, usecols=["compound_id"])["compound_id"].astype(str))
    query_index = {
        (str(row.source_file), str(row.spectrum_id)): int(position)
        for position, row in query_manifest.reset_index(drop=True).iterrows()
    }
    ref_ik14 = ref_manifest.inchikey.map(ik14).to_numpy()
    ref_formula_by_ik14 = {}
    for _, row in ref_manifest.iterrows():
        key = ik14(row.inchikey)
        if key not in ref_formula_by_ik14:
            ref_formula_by_ik14[key] = formula(row.get("smiles", ""))

    candidate_rows = []
    auto_seeds = []
    auto_seed_audit = []
    for hit in matches.itertuples(index=False):
        qidx = query_index.get((str(hit.source_file), str(hit.spectrum_id)))
        if qidx is None:
            raise RuntimeError(f"query absent from embedding cache: {hit.spectrum_id}")
        truth = str(panel.loc[hit.metabolite, "truth_ik14"])
        reference_rows = np.asarray(
            sorted(candidate_map[candidate_map.target_metabolite.eq(hit.metabolite)].library_record_index.unique()),
            dtype=int,
        )
        scores = np.asarray(ref_embeddings[reference_rows] @ query_embeddings[qidx], dtype=float)
        query_id = f"{hit.source_file}|{hit.spectrum_id}"
        raw = pd.DataFrame(
            {
                "candidate_id": ref_ik14[reference_rows],
                "spectral_score": scores,
            }
        )
        collapsed = raw.groupby("candidate_id", as_index=False)["spectral_score"].max()
        if truth not in set(collapsed.candidate_id):
            raise RuntimeError(f"truth absent after candidate collapse: {hit.metabolite}")
        collapsed["query_id"] = query_id
        collapsed["truth_candidate_id"] = truth
        collapsed["truth_formula"] = ref_formula_by_ik14.get(truth, "")
        collapsed["metabolite"] = hit.metabolite
        candidate_rows.append(collapsed)

    candidates = pd.concat(candidate_rows, ignore_index=True)
    # Candidates outside Rhea remain in the ranking with zero network evidence.
    # Removing them would silently make the benchmark easier.
    published_seeds = (
        panel.reset_index()[["metabolite", "truth_ik14"]]
        .drop_duplicates("truth_ik14")
        .rename(columns={"truth_ik14": "seed_compound_id"})
    )
    published_seeds = published_seeds[published_seeds.seed_compound_id.isin(graph_compounds)].copy()
    published_seeds["seed_query_id"] = "published:" + published_seeds["metabolite"].astype(str)
    published_seeds["seed_score"] = 1.0
    published_seeds["reference_kind"] = "evaluation_only_published_identity_leave_target_out"
    published_seeds = published_seeds[
        ["seed_query_id", "seed_compound_id", "seed_score", "reference_kind"]
    ]
    # Build deployment-style seeds from the complete phenotype-blind QC MS/MS
    # pool, not merely the 36 known-target evaluation spectra.  The precursor
    # gate, score threshold and margin threshold are frozen before evaluating
    # the known-target panel.  Formal evaluation still removes the current
    # query and *all* seeds carrying its truth identity.
    ref_precursor = pd.to_numeric(ref_manifest["precursor_mz"], errors="coerce").to_numpy(float)
    for qidx, query in query_manifest.reset_index(drop=True).iterrows():
        query_mz = float(query.precursor_mz)
        if not np.isfinite(query_mz) or query_mz <= 0:
            auto_seed_audit.append(
                {
                    "seed_query_id": f"{query.source_file}|{query.spectrum_id}",
                    "top_candidate_id": "",
                    "top_score": np.nan,
                    "top_margin": np.nan,
                    "candidate_count": 0,
                    "top_candidate_in_graph": False,
                    "passes_frozen_absolute_gate": False,
                    "rejection_reason": "invalid_precursor_mz",
                }
            )
            continue
        ppm_error = np.abs(ref_precursor - query_mz) / query_mz * 1e6
        reference_rows = np.flatnonzero(np.isfinite(ppm_error) & (ppm_error <= args.ppm_tolerance))
        if reference_rows.size == 0:
            auto_seed_audit.append(
                {
                    "seed_query_id": f"{query.source_file}|{query.spectrum_id}",
                    "top_candidate_id": "",
                    "top_score": np.nan,
                    "top_margin": np.nan,
                    "candidate_count": 0,
                    "top_candidate_in_graph": False,
                    "passes_frozen_absolute_gate": False,
                    "rejection_reason": "no_strict10ppm_candidate",
                }
            )
            continue
        scores = np.asarray(ref_embeddings[reference_rows] @ query_embeddings[int(qidx)], dtype=float)
        collapsed = (
            pd.DataFrame({"candidate_id": ref_ik14[reference_rows], "spectral_score": scores})
            .groupby("candidate_id", as_index=False)["spectral_score"].max()
            .sort_values(["spectral_score", "candidate_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
        top = collapsed.iloc[0]
        second = float(collapsed.iloc[1].spectral_score) if len(collapsed) > 1 else -1.0
        margin = float(top.spectral_score) - second
        top_in_graph = str(top.candidate_id) in graph_compounds
        passes_absolute = bool(
            float(top.spectral_score) >= args.auto_seed_minimum_score
            and margin >= args.auto_seed_minimum_margin
            and top_in_graph
        )
        rejection_reason = (
            "accepted"
            if passes_absolute
            else "top_candidate_outside_graph"
            if not top_in_graph
            else "score_below_threshold"
            if float(top.spectral_score) < args.auto_seed_minimum_score
            else "margin_below_threshold"
        )
        auto_seed_audit.append(
            {
                "seed_query_id": f"{query.source_file}|{query.spectrum_id}",
                "top_candidate_id": str(top.candidate_id),
                "top_score": float(top.spectral_score),
                "top_margin": margin,
                "candidate_count": int(len(collapsed)),
                "top_candidate_in_graph": bool(top_in_graph),
                "passes_frozen_absolute_gate": passes_absolute,
                "rejection_reason": rejection_reason,
            }
        )
        if passes_absolute:
            auto_seeds.append(
                {
                    "seed_query_id": f"{query.source_file}|{query.spectrum_id}",
                    "seed_compound_id": str(top.candidate_id),
                    "seed_score": float(np.clip(top.spectral_score, 0, 1)),
                    "reference_kind": "deployable_qc_dreams_top1_strict10ppm_margin",
                }
            )
    auto_seed_table = (
        pd.DataFrame(auto_seeds, columns=published_seeds.columns)
        .drop_duplicates(["seed_query_id", "seed_compound_id"])
    )

    candidate_path = out / "candidates.csv.gz"
    auto_path = out / "seeds_auto.csv"
    auto_audit_path = out / "auto_seed_audit.csv.gz"
    published_path = out / "seeds_published_leave_target_out.csv"
    candidates.to_csv(candidate_path, index=False)
    auto_seed_table.to_csv(auto_path, index=False)
    pd.DataFrame(auto_seed_audit).to_csv(auto_audit_path, index=False)
    published_seeds.to_csv(published_path, index=False)
    baseline_correct = []
    for _, group in candidates.groupby("query_id", sort=False):
        top_score = float(group.spectral_score.max())
        top = group[np.isclose(group.spectral_score, top_score, atol=1e-12, rtol=0)]
        baseline_correct.append(
            len(top) == 1 and str(top.iloc[0].candidate_id) == str(group.iloc[0].truth_candidate_id)
        )
    report = {
        "status": "mtbls1905_bioaware_benchmark_input_complete",
        "queries": int(candidates.query_id.nunique()),
        "targets": int(candidates.metabolite.nunique()),
        "candidate_rows": int(len(candidates)),
        "baseline_recall1": float(np.mean(baseline_correct)),
        "candidate_graph_coverage": float(candidates.candidate_id.isin(graph_compounds).mean()),
        "truth_graph_coverage": float(
            candidates.groupby("query_id").first().truth_candidate_id.isin(graph_compounds).mean()
        ),
        "auto_seed_rows": int(len(auto_seed_table)),
        "auto_seed_compounds": int(auto_seed_table.seed_compound_id.nunique()),
        "auto_seed_audit_rows": int(len(auto_seed_audit)),
        "published_seed_rows": int(len(published_seeds)),
        "phenotype_labels_used": False,
        "seed_regimes": {
            "auto": "deployable phenotype-blind QC spectral Top-1 under strict precursor, score and margin gates",
            "published_leave_target_out": "evaluation-only graph headroom; evaluator must exclude held-out truth identity",
        },
        "auto_seed_parameters": {
            "minimum_score": float(args.auto_seed_minimum_score),
            "minimum_margin": float(args.auto_seed_minimum_margin),
            "ppm_tolerance": float(args.ppm_tolerance),
        },
        "provenance": {str(path): sha256(path) for path in required_paths if path.is_file()},
        "artifacts": {
            "candidates": str(candidate_path),
            "candidates_sha256": sha256(candidate_path),
            "auto_seeds": str(auto_path),
            "auto_seeds_sha256": sha256(auto_path),
            "auto_seed_audit": str(auto_audit_path),
            "auto_seed_audit_sha256": sha256(auto_audit_path),
            "published_seeds": str(published_path),
            "published_seeds_sha256": sha256(published_path),
        },
        "claim_limit": "Known-target external retrieval benchmark. It cannot establish additional cohort annotations or disease mechanism.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
