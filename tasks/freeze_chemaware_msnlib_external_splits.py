"""Freeze score-blind ChemAware MSnLib development/confirmation/test splits.

Queries are grouped before splitting whenever they share an identity, formula,
non-empty Murcko scaffold, a candidate reference spectrum, or a highly similar
truth structure.  This prevents direct candidate and close-chemistry leakage.
No model score is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("discovery", "confirmation", "final_test")
TARGET_FRACTIONS = np.asarray([0.40, 0.30, 0.30], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def union_groups(uf: UnionFind, groups: Iterable[Iterable[int]]) -> None:
    for group in groups:
        if isinstance(group, tuple) and len(group) == 2:
            group = group[1]
        values = [int(value) for value in group]
        for value in values[1:]:
            uf.union(values[0], value)


def component_tie_order(members: list[int]) -> int:
    token = ",".join(map(str, sorted(members))).encode()
    return int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=ROOT / "data/validation/chemaware_msnlib_external_manifest_v2",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/chemaware_msnlib_external_benchmark_v1",
    )
    parser.add_argument("--tanimoto-threshold", type=float, default=.80)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")

    query_path = args.manifest_dir / "queries.csv.gz"
    edge_path = args.manifest_dir / "candidate_edges.csv.gz"
    reference_path = args.manifest_dir / "reference_library.csv.gz"
    source_report_path = args.manifest_dir / "report.json"
    queries = pd.read_csv(query_path).sort_values("query_id").reset_index(drop=True)
    edges = pd.read_csv(edge_path)
    references = pd.read_csv(reference_path)
    if not np.array_equal(queries["query_id"].to_numpy(), np.arange(len(queries))):
        raise RuntimeError("query_id must be contiguous and row-aligned")
    if edges.groupby("query_id")["label"].sum().ne(1).any():
        raise RuntimeError("each source query must have exactly one positive")

    queries["formula"] = queries["formula"].fillna("").astype(str)
    queries["murcko_scaffold"] = queries["murcko_scaffold"].fillna("").astype(str)
    queries["primary_panel"] = (
        queries["same_merge_type_as_true_reference"].astype(bool)
        & queries["adduct"].isin(["[M+H]+", "[M-H]-"])
    )
    queries["primary_same_formula_hard"] = (
        queries["primary_panel"] & (queries["same_formula_negative_identities"] > 0)
    )

    uf = UnionFind(len(queries))
    union_groups(uf, queries.groupby("ik14")["query_id"])
    union_groups(uf, queries.groupby("formula")["query_id"])
    nonempty_scaffold = queries.loc[queries["murcko_scaffold"] != ""]
    union_groups(uf, nonempty_scaffold.groupby("murcko_scaffold")["query_id"])
    union_groups(uf, edges.groupby("reference_role_id")["query_id"].unique())

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for query_id, smiles in zip(queries["query_id"], queries["canonical_smiles"]):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise RuntimeError(f"invalid query structure at query_id={query_id}")
        fingerprints.append(generator.GetFingerprint(mol))
    similar_pairs = 0
    for left, fingerprint in enumerate(fingerprints[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[left + 1:])
        for offset, similarity in enumerate(similarities, start=left + 1):
            if similarity >= args.tanimoto_threshold:
                uf.union(left, offset)
                similar_pairs += 1

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for query_id in range(len(queries)):
        members_by_root[uf.find(query_id)].append(query_id)
    components = list(members_by_root.values())

    def vector(members: list[int]) -> np.ndarray:
        frame = queries.iloc[members]
        primary = frame["primary_panel"].astype(bool)
        return np.asarray([
            len(frame),
            int(primary.sum()),
            int(frame["primary_same_formula_hard"].sum()),
            int((primary & (frame["adduct"] == "[M+H]+")).sum()),
            int((primary & (frame["adduct"] == "[M-H]-")).sum()),
        ], dtype=np.float64)

    component_vectors = {tuple(members): vector(members) for members in components}
    totals = sum(component_vectors.values(), start=np.zeros(5, dtype=np.float64))
    targets = TARGET_FRACTIONS[:, None] * totals[None, :]
    assigned = np.zeros((len(SPLITS), 5), dtype=np.float64)
    split_members: dict[str, list[int]] = {name: [] for name in SPLITS}
    ordered = sorted(
        components,
        key=lambda members: (
            -component_vectors[tuple(members)][1],
            -component_vectors[tuple(members)][0],
            component_tie_order(members),
        ),
    )
    for members in ordered:
        value = component_vectors[tuple(members)]
        scores = []
        for split_index in range(len(SPLITS)):
            trial = assigned.copy()
            trial[split_index] += value
            scaled_error = (trial - targets) / np.maximum(targets, 1.0)
            overflow = np.maximum(trial - 1.12 * targets, 0.0) / np.maximum(targets, 1.0)
            scores.append(float(np.square(scaled_error).sum() + 4.0 * np.square(overflow).sum()))
        best = min(range(len(SPLITS)), key=lambda index: (scores[index], index))
        assigned[best] += value
        split_members[SPLITS[best]].extend(members)

    split_by_query = {}
    for split, members in split_members.items():
        for query_id in members:
            split_by_query[query_id] = split
    queries["split"] = queries["query_id"].map(split_by_query)
    edges["split"] = edges["query_id"].map(split_by_query)
    if queries["split"].isna().any() or edges["split"].isna().any():
        raise RuntimeError("incomplete split assignment")

    # Leakage checks on query truth groups and exact candidate spectra.
    def cross_split_count(column: str, frame: pd.DataFrame = queries) -> int:
        return int((frame.groupby(column)["split"].nunique() > 1).sum())

    nonempty = queries.loc[queries["murcko_scaffold"] != ""]
    candidate_split = edges[["reference_role_id", "split"]].drop_duplicates()
    leakage = {
        "query_ik14_across_splits": cross_split_count("ik14"),
        "query_formula_across_splits": cross_split_count("formula"),
        "nonempty_query_murcko_scaffold_across_splits": cross_split_count(
            "murcko_scaffold", nonempty
        ),
        "candidate_reference_spectrum_across_splits": int(
            (candidate_split.groupby("reference_role_id")["split"].nunique() > 1).sum()
        ),
        "tanimoto_ge_threshold_query_pairs_across_splits": 0,
    }
    for left, fingerprint in enumerate(fingerprints[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[left + 1:])
        for right, similarity in enumerate(similarities, start=left + 1):
            if (
                similarity >= args.tanimoto_threshold
                and queries.at[left, "split"] != queries.at[right, "split"]
            ):
                leakage["tanimoto_ge_threshold_query_pairs_across_splits"] += 1
    if any(leakage.values()):
        raise RuntimeError(f"split leakage detected: {leakage}")

    split_counts = {}
    for split in SPLITS:
        frame = queries.loc[queries["split"] == split]
        primary = frame.loc[frame["primary_panel"]]
        hard = frame.loc[frame["primary_same_formula_hard"]]
        split_counts[split] = {
            "queries": len(frame),
            "primary_queries": len(primary),
            "primary_same_formula_hard_queries": len(hard),
            "primary_mh_queries": int((primary["adduct"] == "[M+H]+").sum()),
            "primary_mminus_h_queries": int((primary["adduct"] == "[M-H]-").sum()),
            "query_ik14": int(frame["ik14"].nunique()),
            "query_formula": int(frame["formula"].nunique()),
            "query_murcko_scaffold": int(frame["murcko_scaffold"].nunique()),
            "candidate_edges": int((edges["split"] == split).sum()),
        }
    for split, counts in split_counts.items():
        if counts["primary_queries"] < 50 or counts["primary_same_formula_hard_queries"] < 35:
            raise RuntimeError(f"minimum primary-panel size failed for {split}: {counts}")
        if counts["primary_mh_queries"] < 20 or counts["primary_mminus_h_queries"] < 15:
            raise RuntimeError(f"minimum adduct stratum failed for {split}: {counts}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    query_out = output / "queries_with_split.csv.gz"
    edge_out = output / "candidate_edges_with_split.csv.gz"
    queries.to_csv(query_out, index=False)
    edges.to_csv(edge_out, index=False)

    final_queries = queries.loc[queries["split"] == "final_test"].copy()
    final_ids = set(final_queries["query_id"].astype(int))
    final_edges = edges.loc[edges["query_id"].isin(final_ids)].copy()
    final_reference_ids = set(final_edges["reference_role_id"].astype(int))
    final_references = references.loc[references["role_id"].isin(final_reference_ids)].copy()
    query_input_columns = [
        "query_id", "file", "line", "raw_file_id", "acquisition_date", "adduct",
        "precursor_mz", "merge_type", "collision_energy", "num_peaks", "spectrum_hash",
        "primary_panel",
    ]
    reference_input_columns = [
        "role_id", "file", "line", "raw_file_id", "acquisition_date", "adduct",
        "precursor_mz", "merge_type", "collision_energy", "num_peaks", "spectrum_hash",
    ]
    test_queries_path = output / "final_test_queries_blinded.csv.gz"
    test_references_path = output / "final_test_references_blinded.csv.gz"
    test_candidates_path = output / "final_test_candidates_blinded.csv.gz"
    test_key_path = output / "final_test_key_private.csv.gz"
    final_queries[query_input_columns].to_csv(test_queries_path, index=False)
    final_references[reference_input_columns].to_csv(test_references_path, index=False)
    final_edges[["query_id", "reference_role_id", "ppm_error"]].to_csv(
        test_candidates_path, index=False
    )
    final_edges[["query_id", "reference_role_id", "label", "same_formula"]].to_csv(
        test_key_path, index=False
    )

    component_sizes = [len(members) for members in components]
    report = {
        "status": "chemaware_msnlib_external_benchmark_split_frozen",
        "formal_training_authorized": False,
        "model_scores_read": False,
        "final_test_consumed": False,
        "primary_panel": (
            "same merge type between query and true reference; [M+H]+ or [M-H]-; "
            "at least two identities within the strict 10-ppm same-adduct window"
        ),
        "secondary_panel": "all mass-competitive manifest queries, including merge mismatch and [M+Na]+",
        "split_contract": {
            "target_fractions": dict(zip(SPLITS, TARGET_FRACTIONS.tolist())),
            "grouping_edges": [
                "same query IK14",
                "same query molecular formula",
                "same non-empty query Murcko scaffold",
                "shared candidate reference spectrum",
                f"query-truth Morgan radius-2 Tanimoto >= {args.tanimoto_threshold:g}",
            ],
            "assignment": "deterministic component-level greedy balance; no model score",
            "minimum_per_split": {
                "primary_queries": 50,
                "primary_same_formula_hard_queries": 35,
                "primary_mh_queries": 20,
                "primary_mminus_h_queries": 15,
            },
        },
        "evaluation_contract": {
            "primary_endpoint": (
                "identity-equal Recall@1 difference versus the frozen official DreaMS baseline "
                "on the primary panel"
            ),
            "strict_top1": "truth score must be strictly greater than every negative; top ties count incorrect",
            "paired_safety": "report corrected and introduced errors separately",
            "uncertainty": "cluster bootstrap over true molecular formula",
            "mandatory_strata": [
                "same-formula-negative present",
                "adduct",
                "query/reference acquisition-date pair",
                "query/reference merge-type pair",
                "candidate-set size",
            ],
            "selection_rule": (
                "discovery may generate hypotheses; confirmation may choose once among frozen "
                "hypotheses; final_test is evaluated once after all code and thresholds are frozen"
            ),
        },
        "counts": {
            "components": len(components),
            "component_size": {
                "min": min(component_sizes),
                "median": float(np.median(component_sizes)),
                "p90": float(np.quantile(component_sizes, .9)),
                "max": max(component_sizes),
            },
            "high_tanimoto_edges_used": similar_pairs,
            "splits": split_counts,
        },
        "leakage_audit": leakage,
        "claim_limit": (
            "This is a future external stress benchmark within the already public MSV000094528 "
            "resource, not proof of a never-before-seen source. Its strongest defensible novelty "
            "is temporal, identity, spectrum-hash, formula, scaffold, high-similarity and exact-"
            "candidate separation under a preregistered post-freeze evaluation."
        ),
        "provenance": {
            "script_sha256": sha256(Path(__file__)),
            "source_report_sha256": sha256(source_report_path),
            "source_queries_sha256": sha256(query_path),
            "source_edges_sha256": sha256(edge_path),
            "source_references_sha256": sha256(reference_path),
            "queries_with_split_sha256": sha256(query_out),
            "candidate_edges_with_split_sha256": sha256(edge_out),
            "final_test_queries_sha256": sha256(test_queries_path),
            "final_test_references_sha256": sha256(test_references_path),
            "final_test_candidates_sha256": sha256(test_candidates_path),
            "final_test_key_sha256": sha256(test_key_path),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
