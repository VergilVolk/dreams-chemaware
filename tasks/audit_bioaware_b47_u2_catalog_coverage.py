#!/usr/bin/env python
"""Truth-blind catalogue-coverage closure for B47 consensus seed identities.

U2 never scores a reaction event.  It asks whether identities that already
passed the frozen spectral and cross-sample consensus gates are represented by
Rhea, strict KEGG, or only by the broader MetDNA2 EMRN expansion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "t", "yes"}


def normalize_ik14(series: pd.Series) -> pd.Series:
    value = series.fillna("").astype(str).str.strip().str.upper().str[:14]
    return value.where(value.str.fullmatch(r"[A-Z]{14}"), "")


def edge_degree(edges: pd.DataFrame, left: str, right: str) -> pd.Series:
    pairs = edges[[left, right]].copy()
    pairs[left] = normalize_ik14(pairs[left])
    pairs[right] = normalize_ik14(pairs[right])
    pairs = pairs[(pairs[left] != "") & (pairs[right] != "")]
    pairs = pairs[pairs[left] != pairs[right]].drop_duplicates()
    mirrored = pd.concat([
        pairs.rename(columns={left: "node", right: "neighbour"}),
        pairs.rename(columns={right: "node", left: "neighbour"}),
    ], ignore_index=True)
    return mirrored.groupby("node")["neighbour"].nunique().astype(int)


def route_decision(rhea_safe: int, exact_union_provisional: int, emrn_union: int) -> dict:
    if rhea_safe >= 200:
        return {
            "code": "RHEA_DENOMINATOR_ELIGIBLE",
            "next_action": "freeze exact Rhea event and structured-null contracts",
            "pass_to_kegg_currency_curation": False,
        }
    if exact_union_provisional >= 200:
        return {
            "code": "STRICT_KEGG_CAN_CLOSE_DENOMINATOR_PROVISIONALLY",
            "next_action": (
                "curate currency/hub status for strict-KEGG-only identities and "
                "freeze source-labelled exact edges before event scoring"
            ),
            "pass_to_kegg_currency_curation": True,
        }
    if emrn_union >= 200:
        return {
            "code": "ONLY_EMRN_EXPANSION_CLOSES_DENOMINATOR",
            "next_action": (
                "do not treat EMRN-expanded links as exact biochemical events; "
                "obtain a broader exact catalogue or independent seed source"
            ),
            "pass_to_kegg_currency_curation": False,
        }
    return {
        "code": "CATALOGUE_COVERAGE_UNDERPOWERED",
        "next_action": "obtain a broader exact catalogue or independent seed source",
        "pass_to_kegg_currency_curation": False,
    }


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--u1c-dir", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--rhea-participants", type=Path, required=True)
    parser.add_argument("--kegg-compounds", type=Path, required=True)
    parser.add_argument("--kegg-edges", type=Path, required=True)
    parser.add_argument("--emrn-compounds", type=Path, required=True)
    parser.add_argument("--emrn-edges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-degree", type=int, default=250)
    args = parser.parse_args()

    paths = {
        "u1c_report": args.u1c_dir / "report.json",
        "u1c_ledger": args.u1c_dir / "query_stage_ledger.csv.gz",
        "seed_scores": args.seed_dir / "candidate_scores.csv.gz",
        "rhea_participants": args.rhea_participants,
        "kegg_compounds": args.kegg_compounds,
        "kegg_edges": args.kegg_edges,
        "emrn_compounds": args.emrn_compounds,
        "emrn_edges": args.emrn_edges,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite U2 audit: {output}")

    u1c = json.loads(paths["u1c_report"].read_text(encoding="utf-8"))
    if (
        u1c.get("status") != "bioaware_b47_u1c_seed_denominator_audit_complete"
        or u1c.get("truth_blind") is not True
        or u1c.get("provenance", {}).get("query_stage_ledger_sha256")
        != sha256_file(paths["u1c_ledger"])
    ):
        raise RuntimeError("U2 requires a valid frozen U1c artifact")

    query = pd.read_csv(paths["u1c_ledger"], dtype={"query_id": str})
    consensus_mask = (
        query["primary_absolute_gate"].astype(bool)
        & query["primary_feature_gate"].fillna(False).astype(bool)
        & query["top_candidate_id"].fillna("").astype(str).eq(
            query["modal_candidate_id"].fillna("").astype(str)
        )
    )
    consensus = query.loc[consensus_mask].copy()
    consensus["candidate_id"] = normalize_ik14(consensus["top_candidate_id"])
    if (consensus["candidate_id"] == "").any():
        raise RuntimeError("consensus ledger contains a malformed candidate identity")
    expected = u1c["stages"]["absolute_and_feature_consensus"]
    if (
        len(consensus) != int(expected["query_events"])
        or consensus["candidate_id"].nunique() != int(expected["candidate_identities"])
    ):
        raise RuntimeError("U2 failed to reproduce the U1c consensus denominator")

    scores = pd.read_csv(
        paths["seed_scores"], usecols=["candidate_id", "candidate_formula"], dtype=str,
    ).drop_duplicates()
    scores["candidate_id"] = normalize_ik14(scores["candidate_id"])
    formula_sets = scores[scores["candidate_id"] != ""].groupby("candidate_id")[
        "candidate_formula"
    ].agg(lambda x: "|".join(sorted(set(x.dropna().astype(str)))))

    rhea = pd.read_csv(
        paths["rhea_participants"],
        usecols=["compound_id", "reaction_id", "is_currency"],
    )
    rhea["compound_id"] = normalize_ik14(rhea["compound_id"])
    rhea = rhea[rhea["compound_id"] != ""].copy()
    rhea_degree = rhea.groupby("compound_id")["reaction_id"].nunique().astype(int)
    rhea_currency = rhea.groupby("compound_id")["is_currency"].agg(
        lambda x: any(parse_bool(value) for value in x)
    )

    kegg_edges = pd.read_csv(paths["kegg_edges"], usecols=["ik14_a", "ik14_b"])
    kegg_degree = edge_degree(kegg_edges, "ik14_a", "ik14_b")
    kegg_compounds = pd.read_csv(
        paths["kegg_compounds"], usecols=["inchikey1", "id", "name", "formula"], dtype=str,
    )
    kegg_compounds["candidate_id"] = normalize_ik14(kegg_compounds["inchikey1"])
    kegg_compounds = kegg_compounds[kegg_compounds["candidate_id"] != ""]
    kegg_metadata = kegg_compounds.groupby("candidate_id").agg(
        kegg_ids=("id", lambda x: "|".join(sorted(set(x.dropna())))),
        kegg_names=("name", lambda x: "|".join(sorted(set(x.dropna())))),
    )

    emrn_edges = pd.read_csv(
        paths["emrn_edges"],
        usecols=["source_ik14", "target_ik14", "minimum_step"],
    )
    emrn_edges["source_ik14"] = normalize_ik14(emrn_edges["source_ik14"])
    emrn_edges["target_ik14"] = normalize_ik14(emrn_edges["target_ik14"])
    emrn_edges = emrn_edges[
        (emrn_edges["source_ik14"] != "") & (emrn_edges["target_ik14"] != "")
    ].copy()
    emrn_minimum = pd.concat([
        emrn_edges[["source_ik14", "minimum_step"]].rename(columns={"source_ik14": "node"}),
        emrn_edges[["target_ik14", "minimum_step"]].rename(columns={"target_ik14": "node"}),
    ], ignore_index=True).groupby("node")["minimum_step"].min()
    # Read and hash the compound table even though coverage is edge-defined.
    emrn_compounds = pd.read_csv(
        paths["emrn_compounds"], usecols=["inchikey1", "min_reaction_step"], dtype=str,
    )
    emrn_compounds["candidate_id"] = normalize_ik14(emrn_compounds["inchikey1"])

    rows = []
    for identity, group in consensus.groupby("candidate_id", sort=True):
        rd = int(rhea_degree.get(identity, 0))
        rc = bool(rhea_currency.get(identity, False))
        kd = int(kegg_degree.get(identity, 0))
        emrn_step = emrn_minimum.get(identity, np.nan)
        rhea_safe = rd > 0 and rd <= args.maximum_degree and not rc
        # A Rhea-known hub remains blocked even if the smaller KEGG graph gives
        # it an artificially low degree.  KEGG-only identities retain
        # provisional status until their currency semantics are curated.
        kegg_provisional = (
            kd > 0 and kd <= args.maximum_degree and not rc
            and rd <= args.maximum_degree
        )
        if rhea_safe:
            coverage_class = "rhea_safe"
        elif rd > 0 and rc:
            coverage_class = "rhea_currency_blocked"
        elif rd > args.maximum_degree:
            coverage_class = "rhea_hub_blocked"
        elif kegg_provisional:
            coverage_class = "strict_kegg_only_provisional"
        elif pd.notna(emrn_step):
            coverage_class = "emrn_extension_only"
        else:
            coverage_class = "no_catalogue_edge"
        rows.append({
            "candidate_id": identity,
            "candidate_formulas": str(formula_sets.get(identity, "")),
            "query_events": int(len(group)),
            "features": int(group["feature_id"].astype(str).nunique()),
            "samples": int(group[["study", "sample"]].drop_duplicates().shape[0]),
            "rhea_degree": rd,
            "rhea_currency": rc,
            "rhea_safe": rhea_safe,
            "strict_kegg_degree": kd,
            "strict_kegg_degree_safe_provisional": kegg_provisional,
            "kegg_ids": str(kegg_metadata.loc[identity, "kegg_ids"]) if identity in kegg_metadata.index else "",
            "kegg_names": str(kegg_metadata.loc[identity, "kegg_names"]) if identity in kegg_metadata.index else "",
            "emrn_minimum_step": float(emrn_step) if pd.notna(emrn_step) else np.nan,
            "coverage_class": coverage_class,
        })
    ledger = pd.DataFrame(rows)
    rhea_safe_count = int(ledger["rhea_safe"].sum())
    exact_union = int((
        ledger["rhea_safe"] | ledger["strict_kegg_degree_safe_provisional"]
    ).sum())
    emrn_union = int((
        ledger["rhea_safe"]
        | ledger["strict_kegg_degree_safe_provisional"]
        | ledger["emrn_minimum_step"].notna()
    ).sum())
    route = route_decision(rhea_safe_count, exact_union, emrn_union)
    indexed = ledger.set_index("candidate_id")
    coverage_by_study = {}
    for study, group in consensus.groupby("study", sort=True):
        identities = sorted(set(group["candidate_id"].astype(str)))
        local = indexed.loc[identities]
        coverage_by_study[str(study)] = {
            "query_events": int(len(group)),
            "identities": int(len(identities)),
            "rhea_safe": int(local["rhea_safe"].sum()),
            "rhea_or_strict_kegg_safe_provisional": int((
                local["rhea_safe"] | local["strict_kegg_degree_safe_provisional"]
            ).sum()),
            "rhea_or_kegg_or_emrn_edge": int((
                local["rhea_safe"]
                | local["strict_kegg_degree_safe_provisional"]
                | local["emrn_minimum_step"].notna()
            ).sum()),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".bioaware_b47_u2_", dir=output.parent))
    try:
        ledger.to_csv(
            staging / "consensus_identity_coverage.csv.gz", index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        report = {
            "status": "bioaware_b47_u2_catalog_coverage_complete",
            "formal": True,
            "truth_blind": True,
            "consensus_denominator": {
                "query_events": int(len(consensus)),
                "identities": int(len(ledger)),
                "features": int(consensus["feature_id"].astype(str).nunique()),
            },
            "coverage": {
                "rhea_covered": int(ledger["rhea_degree"].gt(0).sum()),
                "rhea_safe": rhea_safe_count,
                "rhea_currency_blocked": int(ledger["rhea_currency"].sum()),
                "rhea_hub_blocked": int(ledger["rhea_degree"].gt(args.maximum_degree).sum()),
                "strict_kegg_covered": int(ledger["strict_kegg_degree"].gt(0).sum()),
                "strict_kegg_degree_safe_provisional": int(
                    ledger["strict_kegg_degree_safe_provisional"].sum()
                ),
                "rhea_or_strict_kegg_safe_provisional": exact_union,
                "rhea_or_kegg_or_emrn_edge": emrn_union,
                "coverage_class_counts": {
                    str(key): int(value)
                    for key, value in ledger["coverage_class"].value_counts().sort_index().items()
                },
                "by_study": coverage_by_study,
            },
            "identity_gate": {
                "required": 200,
                "rhea_shortfall": max(0, 200 - rhea_safe_count),
                "provisional_exact_union_shortfall": max(0, 200 - exact_union),
            },
            "fixed_route_decision": route,
            "gates": {
                "reproduces_u1c_consensus_238_identities": len(ledger) == 238,
                "reproduces_u1c_rhea_safe_127_identities": rhea_safe_count == 127,
                "strict_kegg_can_provisionally_close_200_identity_gate": exact_union >= 200,
                "emrn_not_used_as_exact_event": True,
            },
            "contracts": {
                "truth_opened": False,
                "phenotype_used": False,
                "model_fitted": False,
                "reaction_event_scored": False,
                "catalogue_membership_used_for_ranking": False,
                "P2b_used": False,
                "strict_kegg_only_currency_status_provisional": True,
            },
            "provenance": {
                **{f"{name}_sha256": sha256_file(path) for name, path in paths.items()},
                "identity_ledger_sha256": sha256_file(
                    staging / "consensus_identity_coverage.csv.gz"
                ),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "parameters": {"maximum_degree": int(args.maximum_degree)},
            "claim_limit": (
                "Truth-blind catalogue coverage headroom only. Strict-KEGG-only "
                "currency safety is provisional, EMRN expansion is not an exact "
                "reaction event, and no annotation or embedding gain is measured."
            ),
        }
        atomic_json(staging / "report.json", report)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
