#!/usr/bin/env python
"""Prepare a phenotype-blind MTBLS13729 BioAware leave-one-seed-out pilot.

The held-out reference is a *high-confidence frozen spectral annotation*, not a
chemical-standard truth label.  The pilot therefore tests whether independent
reaction neighbors add reproducible support; it cannot establish structure or
biology by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


FORBIDDEN_PHENOTYPE = re.compile(
    r"(^|_)(rmu|rtu|rn|ln|tumou?r|normal|case|control|phenotype|disease|histology|tissue|group_?label|fold_?change|log2fc)(_|$)",
    re.I,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formula_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles)) if smiles and str(smiles) != "nan" else None
    return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--link-dir", type=Path, default=Path("data/mtbls13729/ms1_ms2_link"))
    parser.add_argument("--participants", type=Path, default=Path("data/reference/bioaware_rhea/rhea_participants.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/bioaware_v1_input"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--minimum-cosine", type=float, default=0.80)
    parser.add_argument("--minimum-support-spectra", type=int, default=2)
    parser.add_argument("--minimum-agreement", type=float, default=0.60)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"fail-closed: output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if not args.participants.exists():
        raise FileNotFoundError(args.participants)
    participants = pd.read_csv(args.participants, usecols=["compound_id"])
    graph_compounds = set(participants["compound_id"].dropna().astype(str))

    candidate_rows = []
    seed_rows = []
    panel_reports = {}
    provenance = {"participants_sha256": sha256(args.participants)}
    for panel in args.panels:
        candidate_path = args.link_dir / f"{panel}__feature_annotation_candidates.csv.gz"
        best_path = args.link_dir / f"{panel}__feature_best_annotations.csv.gz"
        if not candidate_path.exists() or not best_path.exists():
            panel_reports[panel] = {
                "status": "missing_inputs", "candidate_path": str(candidate_path), "best_path": str(best_path)
            }
            continue
        candidates = pd.read_csv(candidate_path)
        best = pd.read_csv(best_path)
        suspicious = sorted(
            column for column in set(candidates.columns) | set(best.columns) if FORBIDDEN_PHENOTYPE.search(str(column))
        )
        if suspicious:
            raise RuntimeError(f"phenotype-like columns are forbidden in BioAware identity input: {suspicious}")
        required_candidates = {"feature_id", "ik14", "max_cosine"}
        required_best = {
            "feature_id", "best_ik14", "best_smiles", "max_cosine", "n_support_spectra",
            "structure_agreement_fraction", "annotation_evidence_tier",
        }
        if not required_candidates <= set(candidates) or not required_best <= set(best):
            raise RuntimeError(
                f"unexpected {panel} schema; candidate missing={sorted(required_candidates-set(candidates))}, "
                f"best missing={sorted(required_best-set(best))}"
            )
        provenance[f"{panel}_candidates_sha256"] = sha256(candidate_path)
        provenance[f"{panel}_best_sha256"] = sha256(best_path)
        best = best.copy()
        best["query_id"] = panel + ":" + best["feature_id"].astype(str)
        best["truth_candidate_id"] = best["best_ik14"].fillna("").astype(str)
        best["truth_formula"] = best["best_smiles"].fillna("").map(formula_from_smiles)
        eligible = best[
            (best["truth_candidate_id"].isin(graph_compounds))
            & (best["max_cosine"] >= args.minimum_cosine)
            & (best["n_support_spectra"] >= args.minimum_support_spectra)
            & (best["structure_agreement_fraction"] >= args.minimum_agreement)
            & (best["annotation_evidence_tier"] == "Level 2a-supported")
        ].copy()
        candidate = candidates.copy()
        candidate["query_id"] = panel + ":" + candidate["feature_id"].astype(str)
        candidate["candidate_id"] = candidate["ik14"].fillna("").astype(str)
        candidate["spectral_score"] = pd.to_numeric(candidate["max_cosine"], errors="coerce")
        candidate = candidate[
            candidate["query_id"].isin(eligible["query_id"])
            & np.isfinite(candidate["spectral_score"])
        ].copy()
        # Network-absent candidates remain in the candidate set and receive
        # zero network support downstream.  Removing them would make the
        # BioAware benchmark artificially easier.
        candidate = candidate.sort_values(
            ["query_id", "spectral_score", "candidate_id"], ascending=[True, False, True]
        ).drop_duplicates(["query_id", "candidate_id"])
        query_counts = candidate.groupby("query_id")["candidate_id"].nunique()
        valid_queries = set(query_counts[query_counts >= 2].index)
        eligible = eligible[eligible["query_id"].isin(valid_queries)].copy()
        candidate = candidate[candidate["query_id"].isin(valid_queries)].copy()
        truth_map = eligible.set_index("query_id")[["truth_candidate_id", "truth_formula"]]
        candidate = candidate.join(truth_map, on="query_id", validate="many_to_one")
        truth_present = candidate.groupby("query_id").apply(
            lambda group: bool((group["candidate_id"] == group["truth_candidate_id"].iloc[0]).any()),
            include_groups=False,
        )
        keep = set(truth_present[truth_present].index)
        candidate = candidate[candidate["query_id"].isin(keep)].copy()
        eligible = eligible[eligible["query_id"].isin(keep)].copy()
        for row in eligible.itertuples(index=False):
            seed_rows.append(
                {
                    "seed_query_id": str(row.query_id),
                    "seed_compound_id": str(row.truth_candidate_id),
                    "seed_score": float(np.clip(row.max_cosine, 0, 1)),
                    "reference_kind": "frozen_level2a_supported_spectral_annotation",
                }
            )
        candidate_rows.append(
            candidate[
                ["query_id", "candidate_id", "spectral_score", "truth_candidate_id", "truth_formula", "feature_id"]
            ]
        )
        panel_reports[panel] = {
            "status": "prepared",
            "high_confidence_graph_seeds_before_candidate_gate": int(len(best[
                (best["truth_candidate_id"].isin(graph_compounds))
                & (best["max_cosine"] >= args.minimum_cosine)
                & (best["n_support_spectra"] >= args.minimum_support_spectra)
                & (best["structure_agreement_fraction"] >= args.minimum_agreement)
                & (best["annotation_evidence_tier"] == "Level 2a-supported")
            ])),
            "evaluation_queries": int(len(eligible)),
            "candidate_rows": int(len(candidate)),
        }

    if not candidate_rows or not seed_rows:
        raise RuntimeError(f"no evaluable phenotype-blind BioAware pilot rows; panels={panel_reports}")
    candidate_table = pd.concat(candidate_rows, ignore_index=True)
    seed_table = pd.DataFrame(seed_rows).drop_duplicates(["seed_query_id", "seed_compound_id"])
    candidate_out = out / "candidates.csv.gz"
    seed_out = out / "seeds.csv.gz"
    candidate_table.to_csv(candidate_out, index=False)
    seed_table.to_csv(seed_out, index=False)
    report = {
        "status": "mtbls13729_bioaware_v1_input_complete",
        "formal": True,
        "phenotype_labels_used": False,
        "reference_truth": "frozen Level 2a-supported spectral annotation; not authentic-standard truth",
        "queries": int(candidate_table["query_id"].nunique()),
        "candidate_rows": int(len(candidate_table)),
        "seed_rows": int(len(seed_table)),
        "seed_compounds": int(seed_table["seed_compound_id"].nunique()),
        "panels": panel_reports,
        "parameters": {
            "minimum_cosine": args.minimum_cosine,
            "minimum_support_spectra": args.minimum_support_spectra,
            "minimum_agreement": args.minimum_agreement,
        },
        "provenance": {
            **provenance,
            "candidate_output_sha256": sha256(candidate_out),
            "seed_output_sha256": sha256(seed_out),
        },
        "claim_limit": "This pilot tests independent reaction-neighbor consistency after hiding the query and its truth identity from seeds. It is not standard-confirmed annotation accuracy and uses no phenotype labels.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
