#!/usr/bin/env python3
"""Audit IBD structure candidates against phenotype-blind ion annotations.

The original candidate universe was mass-filtered as if every query were
[M+H]+.  GNPS feature networking supplies an explicit best-ion and neutral
mass for a subset of queries.  Where those fields exist, only candidates whose
exact mass is compatible with the neutral mass are chemically admissible.
Queries lacking an ion/neutral-mass assignment remain adduct-unresolved rather
than being assumed protonated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_ibd200_candidate_context_evidence import candidate_axis


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compatibility_status(neutral_mass: float, exact_mass: float, ppm: float) -> tuple[str, float]:
    if not np.isfinite(neutral_mass) or neutral_mass <= 0:
        return "adduct_unresolved", np.nan
    delta = abs(exact_mass - neutral_mass) / neutral_mass * 1e6
    return ("explicit_neutral_compatible" if delta <= ppm else "explicit_neutral_incompatible"), delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-dir", required=True, type=Path)
    parser.add_argument("--gnps-quant", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--neutral-ppm", type=float, default=10.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    queries_path = args.slice_dir / "query_entities.tsv.gz"
    candidates_path = args.slice_dir / "candidate_structures.tsv.gz"
    score_path = args.slice_dir / "wse_candidate_scores.npz"
    for path in (queries_path, candidates_path, score_path, args.gnps_quant):
        if not path.is_file():
            raise FileNotFoundError(path)

    query_ptr, query_ids, candidate_ids = candidate_axis(score_path)
    edge_queries = np.repeat(query_ids, np.diff(query_ptr))
    queries = pd.read_csv(queries_path, sep="\t", dtype={"raw_entity_id": str})
    quant = pd.read_csv(args.gnps_quant, low_memory=False)
    quant["raw_entity_id"] = quant["row ID"].astype(str)
    ion = queries[["entity_id", "raw_entity_id", "algorithm_answerable"]].merge(
        quant[["raw_entity_id", "row m/z", "row retention time", "best ion", "neutral M mass"]],
        on="raw_entity_id", how="left", validate="one_to_one",
    )
    if set(ion["entity_id"].astype(str)) != set(query_ids):
        raise RuntimeError("quantification rows differ from candidate query axis")

    candidates = pd.read_csv(candidates_path, sep="\t")
    axis = pd.DataFrame({"entity_id": edge_queries, "candidate_id": candidate_ids})
    candidates = axis.merge(
        candidates[["entity_id", "candidate_id", "exact_mass"]],
        on=["entity_id", "candidate_id"], how="left", validate="one_to_one",
    ).merge(
        ion[["entity_id", "best ion", "neutral M mass"]],
        on="entity_id", how="left", validate="many_to_one",
    )
    if candidates["exact_mass"].isna().any():
        raise RuntimeError("candidate exact masses do not cover the score axis")
    classified = [
        compatibility_status(float(neutral), float(exact), args.neutral_ppm)
        for neutral, exact in zip(candidates["neutral M mass"], candidates["exact_mass"])
    ]
    candidates["adduct_compatibility_status"] = [row[0] for row in classified]
    candidates["neutral_mass_ppm"] = [row[1] for row in classified]

    query_summary = (
        candidates.groupby("entity_id", as_index=False)
        .agg(
            candidate_edges=("candidate_id", "size"),
            explicit_neutral_compatible_candidates=(
                "adduct_compatibility_status",
                lambda values: int((values == "explicit_neutral_compatible").sum()),
            ),
            explicit_neutral_incompatible_candidates=(
                "adduct_compatibility_status",
                lambda values: int((values == "explicit_neutral_incompatible").sum()),
            ),
            adduct_unresolved_candidates=(
                "adduct_compatibility_status", lambda values: int((values == "adduct_unresolved").sum()),
            ),
        )
    ).merge(ion, on="entity_id", how="right", validate="one_to_one")
    for column in (
        "candidate_edges", "explicit_neutral_compatible_candidates",
        "explicit_neutral_incompatible_candidates", "adduct_unresolved_candidates",
    ):
        query_summary[column] = query_summary[column].fillna(0).astype(int)
    query_summary["strict_adduct_evaluable"] = (
        query_summary["explicit_neutral_compatible_candidates"] > 0
    )

    score = np.load(score_path, allow_pickle=False)
    wse = np.asarray(score["WSE_score"], dtype=np.float64)
    top_rows = []
    for query_index, query_id in enumerate(query_ids):
        start, stop = int(query_ptr[query_index]), int(query_ptr[query_index + 1])
        finite = np.flatnonzero(np.isfinite(wse[start:stop]))
        if not len(finite):
            continue
        local_candidates = candidate_ids[start:stop]
        order = np.lexsort((local_candidates[finite], -wse[start:stop][finite]))
        edge_index = start + int(finite[order[0]])
        top_rows.append(edge_index)
    top_status = candidates.iloc[top_rows]["adduct_compatibility_status"] if top_rows else pd.Series(dtype=str)

    args.output_dir.mkdir(parents=True)
    edge_path = args.output_dir / "candidate_adduct_compatibility.tsv.gz"
    query_path = args.output_dir / "query_adduct_eligibility.tsv.gz"
    compression = {"method": "gzip", "mtime": 0}
    candidates.to_csv(edge_path, sep="\t", index=False, compression=compression)
    query_summary.to_csv(query_path, sep="\t", index=False, compression=compression)

    explicit_query = query_summary["neutral M mass"].notna()
    incompatible_edges = int(candidates["adduct_compatibility_status"].eq("explicit_neutral_incompatible").sum())
    unresolved_edges = int(candidates["adduct_compatibility_status"].eq("adduct_unresolved").sum())
    candidate_axis_verified = incompatible_edges == 0 and unresolved_edges == 0
    report = {
        "schema": "ibd200_candidate_adduct_compatibility_audit_v1",
        "status": (
            "ADDUCT_AWARE_CANDIDATE_UNIVERSE_VERIFIED"
            if candidate_axis_verified
            else "ORIGINAL_PROTONATED_CANDIDATE_UNIVERSE_NOT_APPLICATION_SAFE"
        ),
        "data_role": "development_consumed",
        "denominators": {
            "queries": int(len(query_summary)),
            "algorithm_answerable_queries_original": int(query_summary["algorithm_answerable"].astype(bool).sum()),
            "queries_with_explicit_ion_and_neutral_mass": int(explicit_query.sum()),
            "queries_with_unresolved_adduct": int((~explicit_query).sum()),
            "queries_with_any_explicit_neutral_compatible_candidate": int(query_summary["strict_adduct_evaluable"].sum()),
            "candidate_edges": int(len(candidates)),
            "explicit_neutral_compatible_edges": int(candidates["adduct_compatibility_status"].eq("explicit_neutral_compatible").sum()),
            "explicit_neutral_incompatible_edges": incompatible_edges,
            "adduct_unresolved_edges": unresolved_edges,
            "WSE_answered_queries": int(len(top_rows)),
            "WSE_top1_explicit_neutral_compatible": int(top_status.eq("explicit_neutral_compatible").sum()),
            "WSE_top1_explicit_neutral_incompatible": int(top_status.eq("explicit_neutral_incompatible").sum()),
            "WSE_top1_adduct_unresolved": int(top_status.eq("adduct_unresolved").sum()),
        },
        "parameters": {"neutral_mass_ppm": float(args.neutral_ppm)},
        "interpretation": (
            "Every candidate-bearing edge is explicitly neutral-mass compatible; queries without an ion "
            "assignment remain zero-candidate abstentions."
            if candidate_axis_verified else
            "The existing candidate axis assumes protonated precursors. Explicit FBMN neutral masses show "
            "that most annotated-query candidates are chemically incompatible; missing ion annotations "
            "cannot be silently treated as [M+H]+. Strict biological identity links must therefore use "
            "only explicit neutral-mass-compatible candidates until an adduct-aware candidate universe is built."
        ),
        "provenance": {
            "query_entities_sha256": sha256(queries_path),
            "candidate_structures_sha256": sha256(candidates_path),
            "WSE_score_bundle_sha256": sha256(score_path),
            "GNPS_quant_sha256": sha256(args.gnps_quant),
            "candidate_audit_sha256": sha256(edge_path),
            "query_audit_sha256": sha256(query_path),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": [
            "FBMN best-ion and neutral-mass fields are phenotype-blind annotations, not definitive adduct truth.",
            "Adduct-unresolved queries are abstentions for strict identity use, not proven incompatible.",
            "This audit invalidates strict biological interpretation of the original candidate universe; it does not invalidate spectrum-only benchmarking on its stated [M+H]+ task.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
