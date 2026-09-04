#!/usr/bin/env python
"""Build an independent KGMN-200STD hidden-seed confirmation manifest.

Only exact, punctuation-insensitive standard-name matches that resolve to one
chemically approved MONA [M-H]- identity are admitted.  Candidate retrieval is
then reconstructed from the frozen official DreaMS embeddings under the same
10 ppm, per-IK14 maximum protocol used by the negative expert.  No BioAware
score or outcome-dependent threshold is computed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


OFFICIAL_CHECKPOINT_SHA256 = "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def strict_unique_top(group: pd.DataFrame, column: str) -> tuple[str, bool]:
    maximum = float(group[column].max())
    top = group[np.isclose(group[column], maximum, rtol=0, atol=1e-12)]
    return str(top.sort_values("candidate_id").iloc[0]["candidate_id"]), len(top) == 1


def balanced_splits(
    identities: list[str], repeats: int, hidden_fraction: float, seed: int
) -> pd.DataFrame:
    identities_array = np.asarray(sorted(set(identities)), dtype=object)
    if len(identities_array) < 2 or not 0 < hidden_fraction < 1:
        raise ValueError("invalid hidden-seed split request")
    rng = np.random.default_rng(seed)
    rng.shuffle(identities_array)
    n_hidden = max(1, int(round(len(identities_array) * hidden_fraction)))
    n_hidden = min(n_hidden, len(identities_array) - 1)
    rows: list[dict] = []
    for repeat in range(repeats):
        positions = (np.arange(n_hidden) + repeat * n_hidden) % len(identities_array)
        hidden = set(identities_array[positions].tolist())
        for identity in identities_array:
            rows.append({
                "repeat": repeat,
                "ik14": str(identity),
                "role": "hidden_validation" if identity in hidden else "seed",
            })
    frame = pd.DataFrame(rows)
    if frame.groupby(["repeat", "ik14"]).size().ne(1).any():
        raise RuntimeError("hidden-seed split duplicated identities")
    hidden_counts = frame[frame["role"].eq("hidden_validation")].groupby("ik14").size()
    if int(hidden_counts.max() - hidden_counts.min()) > 1:
        raise RuntimeError("hidden-seed exposure is unbalanced")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--truth", type=Path,
        default=Path("third_party/MetDNA2/inst/extdata/annotation_initial.csv"),
    )
    parser.add_argument(
        "--query-embeddings", type=Path,
        default=Path(
            "data/validation/kgmn_200std_official_dreams_local_20260901/"
            "official_200std_embeddings.csv.gz"
        ),
    )
    parser.add_argument(
        "--library-manifest", type=Path,
        default=Path("data/models/mona_neg_dreams_emb/manifest.csv"),
    )
    parser.add_argument(
        "--library-embeddings", type=Path,
        default=Path("data/models/mona_neg_dreams_emb/embeddings.npy"),
    )
    parser.add_argument(
        "--approved-library-rows", type=Path,
        default=Path(
            "data/validation/mona_negative_library_chemical_integrity_v1/"
            "approved_m_h_library_rows.npy"
        ),
    )
    parser.add_argument(
        "--frozen-expert", type=Path,
        default=Path(
            "data/validation/bioaware_metdna3_negative_network_expert_v2_chemically_filtered/"
            "artifact.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_kgmn200std_confirmation_manifest_v2"),
    )
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--hidden-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    required = [
        args.truth, args.query_embeddings, args.library_manifest,
        args.library_embeddings, args.approved_library_rows, args.frozen_expert,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")

    truth = pd.read_csv(args.truth)
    needed_truth = {"name", "mz", "rt", "id", "adduct", "formula", "cpd_name"}
    if missing := needed_truth - set(truth.columns):
        raise RuntimeError(f"truth table misses columns: {sorted(missing)}")
    truth = truth[truth["adduct"].astype(str).eq("[M-H]-")].copy()
    truth["normalized_name"] = truth["cpd_name"].map(normalize_name)

    manifest = pd.read_csv(args.library_manifest)
    approved = np.load(args.approved_library_rows, allow_pickle=False).astype(np.int64)
    if approved.ndim != 1 or len(np.unique(approved)) != len(approved):
        raise RuntimeError("approved library rows must be a unique 1D vector")
    if approved.min() < 0 or approved.max() >= len(manifest):
        raise RuntimeError("approved library row is out of range")
    library = manifest.iloc[approved].copy().reset_index(drop=True)
    library["library_row"] = approved
    library["candidate_id"] = library["inchikey"].astype(str).str[:14].str.upper()
    library["normalized_name"] = library["name"].map(normalize_name)
    name_to_identities = library.groupby("normalized_name")["candidate_id"].agg(
        lambda values: sorted(set(map(str, values)))
    )
    truth["mapped_identities"] = truth["normalized_name"].map(name_to_identities)
    truth["truth_candidate_id"] = truth["mapped_identities"].map(
        lambda values: values[0] if isinstance(values, list) and len(values) == 1 else ""
    )

    query_frame = pd.read_csv(args.query_embeddings)
    if query_frame["feature_name"].duplicated().any():
        raise RuntimeError("query embedding names are duplicated")
    embedding_columns = [column for column in query_frame if column.startswith("z_")]
    query_vectors = query_frame.set_index("feature_name")[embedding_columns]
    library_vectors = np.load(args.library_embeddings, mmap_mode="r")
    if len(library_vectors) != len(manifest):
        raise RuntimeError("library manifest/embedding length mismatch")
    approved_vectors = np.asarray(library_vectors[approved], dtype=np.float32)
    vector_norms = np.linalg.norm(approved_vectors, axis=1)
    if np.max(np.abs(vector_norms - 1.0)) > 2e-5:
        raise RuntimeError("approved library embeddings are not unit normalized")

    library_mz = pd.to_numeric(library["precursor_mz"], errors="coerce").to_numpy(float)
    feature_rows: list[dict] = []
    candidate_rows: list[dict] = []
    mapping_rows: list[dict] = []
    for feature_name, group in truth.groupby("name", sort=True):
        identities = sorted(set(group["truth_candidate_id"].astype(str)) - {""})
        reason = ""
        if len(identities) != 1:
            reason = "truth_name_not_uniquely_mapped_to_one_approved_ik14"
        elif feature_name not in query_vectors.index:
            reason = "query_embedding_missing"
        mz = float(pd.to_numeric(group["mz"], errors="raise").iloc[0])
        window = np.flatnonzero(np.abs(library_mz - mz) / mz * 1e6 <= args.ppm)
        candidate_ids = set(library.iloc[window]["candidate_id"].astype(str))
        truth_identity = identities[0] if len(identities) == 1 else ""
        truth_in_mass_window = bool(truth_identity and truth_identity in candidate_ids)
        if not reason and not truth_in_mass_window:
            reason = "truth_not_in_approved_mass_window"
        seed_eligible = bool(not reason or reason == "fewer_than_two_candidate_molecules")
        if not reason and len(candidate_ids) < 2:
            reason = "fewer_than_two_candidate_molecules"
            seed_eligible = True
        mapping_rows.append({
            "feature_name": feature_name,
            "mz": mz,
            "rt": float(pd.to_numeric(group["rt"], errors="raise").iloc[0]),
            "truth_candidate_id": truth_identity,
            "truth_formula": str(group["formula"].iloc[0]),
            "truth_standard_ids": "|".join(sorted(set(group["id"].astype(str)))),
            "truth_names": "|".join(sorted(set(group["cpd_name"].astype(str)))),
            "candidate_molecules": len(candidate_ids),
            "truth_in_mass_window": truth_in_mass_window,
            "seed_eligible": seed_eligible,
            "evaluable": not bool(reason),
            "exclusion_reason": reason,
        })
        if reason:
            continue
        local = library.iloc[window].copy()
        query = query_vectors.loc[feature_name].to_numpy(dtype=np.float32)
        local["spectral_score"] = approved_vectors[window] @ query
        best = local.sort_values(
            ["candidate_id", "spectral_score", "library_row"],
            ascending=[True, False, True], kind="stable",
        ).drop_duplicates("candidate_id", keep="first")
        for row in best.itertuples(index=False):
            candidate_rows.append({
                "query_id": str(feature_name),
                "candidate_id": str(row.candidate_id),
                "spectral_score": float(row.spectral_score),
                "best_library_row": int(row.library_row),
                "reference_spectra": int((local["candidate_id"] == row.candidate_id).sum()),
            })
        local_candidates = pd.DataFrame(candidate_rows)[
            pd.DataFrame(candidate_rows)["query_id"].eq(str(feature_name))
        ]
        baseline, unique = strict_unique_top(local_candidates, "spectral_score")
        feature_rows.append({
            "query_id": str(feature_name), "feature_name": str(feature_name),
            "mz": mz, "rt": float(group["rt"].iloc[0]),
            "truth_candidate_id": truth_identity,
            "truth_formula": str(group["formula"].iloc[0]),
            "baseline_candidate_id": baseline,
            "baseline_unique": unique,
            "baseline_correct": unique and baseline == truth_identity,
            "candidate_molecules": int(len(local_candidates)),
        })
    candidates = pd.DataFrame(candidate_rows)
    queries = pd.DataFrame(feature_rows)
    mappings = pd.DataFrame(mapping_rows)
    if queries.empty or candidates.empty:
        raise RuntimeError("KGMN mapping produced no evaluable queries")
    if candidates.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("candidate table has duplicate query/candidate rows")
    if queries["truth_candidate_id"].nunique() < 20:
        raise RuntimeError("fewer than 20 evaluable truth identities")

    seed_features = mappings[
        mappings["seed_eligible"].astype(bool)
        & mappings["truth_candidate_id"].astype(str).ne("")
    ][[
        "feature_name", "mz", "rt", "truth_candidate_id", "truth_formula"
    ]].rename(columns={"truth_candidate_id": "ik14"}).copy()
    if seed_features["ik14"].nunique() < queries["truth_candidate_id"].nunique():
        raise RuntimeError("seed universe lost an evaluable identity")
    splits = balanced_splits(
        seed_features["ik14"].astype(str).unique().tolist(),
        args.repeats, args.hidden_fraction, args.seed,
    )
    args.output_dir.mkdir(parents=True)
    paths = {
        "queries": args.output_dir / "queries.csv.gz",
        "candidates": args.output_dir / "candidate_scores.csv.gz",
        "mapping": args.output_dir / "truth_mapping_audit.csv.gz",
        "seed_features": args.output_dir / "seed_features.csv.gz",
        "splits": args.output_dir / "hidden_seed_splits.csv.gz",
    }
    queries.to_csv(paths["queries"], index=False, compression="gzip")
    candidates.to_csv(paths["candidates"], index=False, compression="gzip")
    mappings.to_csv(paths["mapping"], index=False, compression="gzip")
    seed_features.to_csv(paths["seed_features"], index=False, compression="gzip")
    splits.to_csv(paths["splits"], index=False, compression="gzip")

    artifact = json.loads(args.frozen_expert.read_text(encoding="utf-8"))
    report = {
        "status": "bioaware_kgmn200std_confirmation_manifest_complete",
        "formal": True,
        "protocol": (
            "exact normalized standard-name to unique chemically approved MONA IK14; [M-H]- only; "
            "strict 10ppm candidates; per-IK14 maximum official DreaMS cosine; >=2 molecules/query"
        ),
        "evaluable_queries": int(len(queries)),
        "evaluable_truth_identities": int(queries["truth_candidate_id"].nunique()),
        "evaluable_truth_formulas": int(queries["truth_formula"].nunique()),
        "candidate_pairs": int(len(candidates)),
        "baseline": {
            "recall1": float(queries["baseline_correct"].mean()),
            "errors": int((~queries["baseline_correct"]).sum()),
            "maximum_possible_delta": float((~queries["baseline_correct"]).mean()),
        },
        "hidden_seed_design": {
            "repeats": args.repeats,
            "hidden_fraction": args.hidden_fraction,
            "identities_per_repeat": int(splits.groupby("repeat")["ik14"].nunique().iloc[0]),
            "seed_eligible_features": int(len(seed_features)),
            "seed_eligible_identities": int(seed_features["ik14"].nunique()),
            "hidden_identities_per_repeat": sorted(
                splits[splits["role"].eq("hidden_validation")].groupby("repeat")["ik14"].nunique().unique().tolist()
            ),
        },
        "frozen_expert": {
            "path": str(args.frozen_expert),
            "sha256": sha256(args.frozen_expert),
            "version": artifact.get("version"),
            "recipe_name": artifact.get("recipe_name"),
        },
        "provenance": {name: sha256(path) for name, path in {
            "truth": args.truth, "query_embeddings": args.query_embeddings,
            "library_manifest": args.library_manifest,
            "library_embeddings": args.library_embeddings,
            "approved_library_rows": args.approved_library_rows,
            **paths,
        }.items()},
        "contracts": {
            "official_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
            "v2_development_queries_used": False,
            "phenotype_used": False,
            "P2b_used": False,
            "outcomes_used_to_choose_mapping": False,
            "truth_identity_removed_from_seed_context_when_evaluated": "enforced in confirmation stage",
        },
        "pass_to_frozen_hidden_seed_confirmation": bool(
            len(queries) >= 40
            and queries["truth_candidate_id"].nunique() >= 20
            and (~queries["baseline_correct"]).sum() >= 5
        ),
        "claim_limit": (
            "This is an artificial standard-mixture mechanistic confirmation manifest, not biological-cohort "
            "generalization and not a BioAware performance result."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
