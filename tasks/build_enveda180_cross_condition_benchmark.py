#!/usr/bin/env python
"""Build a frozen cross-collision-energy Enveda-180 retrieval benchmark.

Construction is model-blind. A high-collision-energy spectrum is the query;
candidate reference spectra come from a different collision energy. Candidate
molecules share the same adduct/polarity and lie within a fixed precursor-mass
window. The positive molecule is materialised first and ties must later count
against it, matching the existing GNPS evaluator contract.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from build_gnps_gold_silver_10ppm_benchmark import expand_pair_ledger
from prepare_enveda180_scoreblind_manifest import open_text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_conflicts(path: Path) -> set[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def ppm_window(value: float, ppm: float) -> tuple[float, float]:
    delta = abs(value) * ppm * 1e-6
    return value - delta, value + delta


def prepare_groups(frame: pd.DataFrame, refs_per_identity: int):
    groups = {}
    for key, group in frame.groupby(["ik14", "adduct"], sort=True):
        group = group.sort_values(["collision_energy", "source_row"], kind="stable")
        energies = group.collision_energy.unique()
        if len(energies) < 2:
            continue
        query = group[group.collision_energy == energies.max()].iloc[0]
        low = group[group.collision_energy == energies.min()].head(refs_per_identity)
        groups[key] = {
            "query_row": int(query.row),
            "query_mz": float(query.precursor_mz),
            "formula": str(query.formula),
            "ref_energy": float(energies.min()),
            "reference_rows": low.row.to_numpy(np.int64),
            "all": group,
        }
    return groups


def candidate_rows_for_group(group: dict, target_energy: float, refs_per_identity: int, query_row: int):
    work = group["all"].copy()
    work = work[work.row != query_row]
    work["energy_distance"] = (work.collision_energy - target_energy).abs()
    return work.sort_values(["energy_distance", "source_row"], kind="stable").head(refs_per_identity).row.to_numpy(np.int64)


def build_panel(
    frame: pd.DataFrame,
    ppm: float,
    max_candidates: int,
    refs_per_identity: int,
    max_queries: int,
    sampling_seed: int,
):
    groups = prepare_groups(frame, refs_per_identity)
    by_adduct = {}
    for adduct in sorted({key[1] for key in groups}):
        entries = [(key, groups[key]["query_mz"]) for key in groups if key[1] == adduct]
        entries.sort(key=lambda item: (item[1], item[0][0]))
        by_adduct[adduct] = (np.asarray([item[1] for item in entries], float), [item[0] for item in entries])

    query_row = []
    query_ik14 = []
    query_formula = []
    query_adduct = []
    query_precursor_mz = []
    query_ptr = [0]
    molecule_ptr = [0]
    molecule_label = []
    molecule_ik14 = []
    molecule_formula = []
    molecule_same_formula = []
    candidate_row = []
    near_query = []
    independent_positive = []

    # A fixed hash order gives a model-blind, chemically broad subset while
    # bounding the one-off construction cost on the 1.17M-spectrum release.
    # It is frozen before any model score is computed.
    query_keys = sorted(
        groups,
        key=lambda item: hashlib.sha256(
            f"{sampling_seed}|{item[0]}|{item[1]}".encode("utf-8")
        ).digest(),
    )
    for key in query_keys:
        truth, adduct = key
        source = groups[key]
        values, keys = by_adduct[adduct]
        low, high = ppm_window(source["query_mz"], ppm)
        left = int(np.searchsorted(values, low, side="left"))
        right = int(np.searchsorted(values, high, side="right"))
        candidate_keys = keys[left:right]
        negatives = [item for item in candidate_keys if item[0] != truth]
        if not negatives:
            continue
        negatives.sort(key=lambda item: (abs(groups[item]["query_mz"] - source["query_mz"]), item[0]))
        selected = [key] + negatives[: max_candidates - 1]
        q_index = int(source["query_row"])
        q_formula = str(source["formula"])
        local_near = False
        complete = True
        local_blocks = []
        for candidate in selected:
            c_group = groups[candidate]
            rows = candidate_rows_for_group(c_group, source["ref_energy"], refs_per_identity, q_index)
            if not len(rows):
                complete = False
                break
            label = candidate[0] == truth
            same_formula = str(c_group["formula"]) == q_formula
            local_near = local_near or (same_formula and not label)
            local_blocks.append((candidate, rows, label, same_formula))
        if not complete or len(local_blocks) < 2:
            continue
        query_row.append(q_index)
        query_ik14.append(truth)
        query_formula.append(q_formula)
        query_adduct.append(adduct)
        query_precursor_mz.append(source["query_mz"])
        near_query.append(local_near)
        independent_positive.append(True)  # distinct collision-energy record
        for candidate, rows, label, same_formula in local_blocks:
            molecule_label.append(label)
            molecule_ik14.append(candidate[0])
            molecule_formula.append(str(groups[candidate]["formula"]))
            molecule_same_formula.append(same_formula)
            candidate_row.extend(rows.tolist())
            molecule_ptr.append(len(candidate_row))
        query_ptr.append(len(molecule_label))
        if len(query_row) % 1000 == 0:
            print(
                f"enveda panel queries={len(query_row):,}/{max_queries:,} "
                f"candidate_molecules={len(molecule_label):,}",
                flush=True,
            )
        if len(query_row) >= max_queries:
            break

    return {
        "query_row": np.asarray(query_row, np.int64),
        "query_ik14": np.asarray(query_ik14, dtype=np.str_),
        "query_formula": np.asarray(query_formula, dtype=np.str_),
        "query_adduct": np.asarray(query_adduct, dtype=np.str_),
        "query_precursor_mz": np.asarray(query_precursor_mz, float),
        "near_query": np.asarray(near_query, bool),
        "independent_positive": np.asarray(independent_positive, bool),
        "query_ptr": np.asarray(query_ptr, np.int64),
        "molecule_ptr": np.asarray(molecule_ptr, np.int64),
        "molecule_label": np.asarray(molecule_label, np.int8),
        "molecule_ik14": np.asarray(molecule_ik14, dtype=np.str_),
        "molecule_formula": np.asarray(molecule_formula, dtype=np.str_),
        "molecule_same_formula": np.asarray(molecule_same_formula, bool),
        "candidate_row": np.asarray(candidate_row, np.int64),
    }


def selected_rows(*panels: dict) -> np.ndarray:
    pieces = []
    for panel in panels:
        pieces.extend([panel["query_row"], panel["candidate_row"]])
    return np.unique(np.concatenate(pieces)).astype(np.int64)


def build_open_set_panel(
    panel: dict,
    match_fraction: float,
    sampling_seed: int,
    namespace: str,
) -> dict:
    """Create a deterministic match/no-match deployment panel.

    For no-match queries the true molecule, including every one of its
    reference spectra, is physically absent from the candidate pool.  The
    choice is identity-hash based and made before model scores exist.
    """
    if not 0.0 < match_fraction < 1.0:
        raise ValueError("open-set match fraction must be in (0, 1)")
    query_fields = (
        "query_row", "query_ik14", "query_formula", "query_adduct", "query_precursor_mz",
        "near_query", "independent_positive",
    )
    molecule_fields = (
        "molecule_label", "molecule_ik14", "molecule_formula",
        "molecule_same_formula",
    )
    result = {name: [] for name in query_fields + molecule_fields}
    result["query_ptr"] = [0]
    result["molecule_ptr"] = [0]
    result["candidate_row"] = []
    result["query_has_match"] = []
    threshold = int(match_fraction * (1 << 64))
    for query in range(len(panel["query_row"])):
        left, right = map(int, panel["query_ptr"][query:query + 2])
        local_labels = np.asarray(panel["molecule_label"][left:right], dtype=bool)
        if int(local_labels.sum()) != 1:
            raise RuntimeError("source panel query does not have exactly one positive")
        identity = str(panel["query_ik14"][query])
        digest = hashlib.sha256(
            f"{sampling_seed}|{namespace}|{identity}".encode("utf-8")
        ).digest()
        has_match = int.from_bytes(digest[:8], "big") < threshold
        # A no-match confidence margin needs at least two remaining molecules.
        if not has_match and int((~local_labels).sum()) < 2:
            has_match = True
        for name in query_fields:
            result[name].append(panel[name][query])
        result["query_has_match"].append(has_match)
        for molecule in range(left, right):
            if not has_match and bool(panel["molecule_label"][molecule]):
                continue
            for name in molecule_fields:
                value = panel[name][molecule]
                if name == "molecule_label" and not has_match:
                    value = False
                result[name].append(value)
            ref_left, ref_right = map(
                int, panel["molecule_ptr"][molecule:molecule + 2]
            )
            result["candidate_row"].extend(
                map(int, panel["candidate_row"][ref_left:ref_right])
            )
            result["molecule_ptr"].append(len(result["candidate_row"]))
        result["query_ptr"].append(len(result["molecule_label"]))
    dtypes = {
        "query_row": np.int64,
        "query_ik14": np.str_,
        "query_formula": np.str_,
        "query_adduct": np.str_,
        "query_precursor_mz": float,
        "near_query": bool,
        "independent_positive": bool,
        "query_has_match": bool,
        "query_ptr": np.int64,
        "molecule_ptr": np.int64,
        "molecule_label": np.int8,
        "molecule_ik14": np.str_,
        "molecule_formula": np.str_,
        "molecule_same_formula": bool,
        "candidate_row": np.int64,
    }
    arrays = {name: np.asarray(values, dtype=dtypes[name]) for name, values in result.items()}
    if int(arrays["query_has_match"].sum()) != int(arrays["molecule_label"].sum()):
        raise RuntimeError("open-set match labels are inconsistent")
    return arrays


def write_curated_mgf(source: Path, selected_source_rows: set[int], target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    active = False
    keep = False
    source_row = -1
    with open_text(source) as reader, target.open("w", encoding="utf-8", newline="\n") as writer:
        for raw in reader:
            line = raw.strip()
            if line == "BEGIN IONS":
                source_row += 1
                active = True
                keep = source_row in selected_source_rows
            if active and keep:
                writer.write(raw if raw.endswith("\n") else raw + "\n")
            if line == "END IONS":
                if keep:
                    count += 1
                active = False
                keep = False
    return count


def remap_panel(panel: dict, mapping: dict[int, int]) -> dict:
    result = {key: np.asarray(value).copy() for key, value in panel.items()}
    result["query_row"] = np.asarray([mapping[int(v)] for v in panel["query_row"]], np.int64)
    result["candidate_row"] = np.asarray([mapping[int(v)] for v in panel["candidate_row"]], np.int64)
    return result


def panel_summary(panel: dict) -> dict:
    summary = {
        "queries": int(len(panel["query_row"])),
        "query_identities": int(len(np.unique(panel["query_ik14"]))),
        "query_formulas": int(len(np.unique(panel["query_formula"]))),
        "candidate_molecules": int(len(panel["molecule_label"])),
        "candidate_spectra": int(len(panel["candidate_row"])),
        "near_queries": int(panel["near_query"].sum()),
        "query_adducts": {
            str(value): int(count)
            for value, count in zip(*np.unique(panel["query_adduct"].astype(str), return_counts=True))
        },
    }
    if "query_has_match" in panel:
        summary.update({
            "match_queries": int(np.asarray(panel["query_has_match"], bool).sum()),
            "no_match_queries": int((~np.asarray(panel["query_has_match"], bool)).sum()),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mgf", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--refs-per-identity", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=25_000)
    parser.add_argument("--sampling-seed", type=int, default=20261008)
    parser.add_argument("--minimum-queries", type=int, default=1000)
    parser.add_argument("--minimum-formula-queries", type=int, default=100)
    parser.add_argument("--open-set-match-fraction", type=float, default=0.70)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite frozen benchmark: {args.out}")

    audit_report = json.loads((args.audit / "report.json").read_text(encoding="utf-8"))
    if audit_report.get("status") != "ENVEDA180_SCOREBLIND_MANIFEST_COMPLETE":
        raise RuntimeError("Enveda score-blind manifest is not complete")
    if audit_report.get("performance_scores_opened") is not False or audit_report.get("model_loaded") is not False:
        raise RuntimeError("Enveda audit is not score blind")
    if audit_report.get("exclusion_policy_complete") is not True:
        raise RuntimeError("Enveda consumed-source exclusion policy is incomplete")
    missing_sources = [
        row.get("name", row.get("path")) for row in audit_report.get("exclusion_sources", [])
        if row.get("required", True) and row.get("status") != "loaded"
    ]
    if missing_sources:
        raise RuntimeError(f"Enveda audit has missing required exclusion sources: {missing_sources}")
    manifest_path = args.audit / "eligible_records.csv.gz"
    conflicts_path = args.audit / "conflicting_spectrum_hashes.txt.gz"
    if sha256(manifest_path) != audit_report.get("manifest_sha256"):
        raise RuntimeError("Enveda audit manifest hash mismatch")
    if sha256(conflicts_path) != audit_report.get("conflicting_hashes_sha256"):
        raise RuntimeError("Enveda conflicting-hash ledger mismatch")
    if sha256(args.source_mgf) != audit_report.get("source_mgf_sha256"):
        raise RuntimeError("Enveda source MGF differs from audited source")
    frame = pd.read_csv(manifest_path, low_memory=False)
    required_columns = {"consumed_identity_overlap", "consumed_formula_overlap", "consumed_spectrum_overlap"}
    if not required_columns.issubset(frame.columns):
        raise RuntimeError(f"Enveda audit lacks leakage columns: {sorted(required_columns - set(frame.columns))}")
    conflicts = load_conflicts(conflicts_path)
    base = frame[
        frame.primary_adduct.astype(bool)
        & frame.primary_adduct_validated.astype(bool)
        & ~frame.consumed_identity_overlap.astype(bool)
        & ~frame.consumed_spectrum_overlap.astype(bool)
        & ~frame.spectrum_hash.astype(str).isin(conflicts)
    ].copy()
    identity = build_panel(
        base, args.ppm, args.max_candidates, args.refs_per_identity,
        args.max_queries, args.sampling_seed,
    )
    formula_base = base[~base.consumed_formula_overlap.astype(bool)].copy()
    formula = build_panel(
        formula_base, args.ppm, args.max_candidates, args.refs_per_identity,
        args.max_queries, args.sampling_seed,
    )
    if len(identity["query_row"]) < args.minimum_queries:
        raise RuntimeError(f"identity panel too small: {len(identity['query_row'])}")
    if len(formula["query_row"]) < args.minimum_formula_queries:
        raise RuntimeError(f"formula panel too small: {len(formula['query_row'])}")

    used = selected_rows(identity, formula)
    selected = frame.set_index("row").loc[used].sort_values("source_row").reset_index()
    mapping = {int(old): new for new, old in enumerate(selected.row.to_numpy(np.int64))}
    selected["audit_row"] = selected["row"]
    selected["row"] = np.arange(len(selected), dtype=np.int64)
    identity = remap_panel(identity, mapping)
    formula = remap_panel(formula, mapping)
    identity_open = build_open_set_panel(
        identity, args.open_set_match_fraction, args.sampling_seed, "identity",
    )
    formula_open = build_open_set_panel(
        formula, args.open_set_match_fraction, args.sampling_seed, "formula",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out / "manifest.csv.gz", index=False)
    written = write_curated_mgf(
        args.source_mgf,
        set(selected.source_row.to_numpy(np.int64).tolist()),
        args.out / "spectra.mgf",
    )
    if written != len(selected):
        raise RuntimeError(f"curated MGF/manifest mismatch: {written} != {len(selected)}")
    for name, panel in (("identity_disjoint", identity), ("formula_disjoint", formula)):
        np.savez_compressed(args.out / f"panel_{name}.npz", **panel)
        np.savez_compressed(args.out / f"pairs_{name}.npz", **expand_pair_ledger(panel))
    for name, panel in (
        ("identity_open_set", identity_open), ("formula_open_set", formula_open),
    ):
        np.savez_compressed(args.out / f"panel_{name}.npz", **panel)
        np.savez_compressed(args.out / f"pairs_{name}.npz", **expand_pair_ledger(panel))

    files = [
        "spectra.mgf", "manifest.csv.gz", "panel_identity_disjoint.npz",
        "panel_formula_disjoint.npz", "pairs_identity_disjoint.npz",
        "pairs_formula_disjoint.npz", "panel_identity_open_set.npz",
        "panel_formula_open_set.npz", "pairs_identity_open_set.npz",
        "pairs_formula_open_set.npz",
    ]
    checksums = {name: sha256(args.out / name) for name in files}
    (args.out / "checksums.sha256").write_text(
        "".join(f"{checksums[name]}  {name}\n" for name in files), encoding="utf-8"
    )
    report = {
        "status": "ENVEDA180_CROSS_CONDITION_BENCHMARK_FROZEN",
        "schema": "enveda180_cross_condition_benchmark_v1",
        "dataset_id": "enveda_180_filtered_20260713",
        "evaluation_role": "sealed_external_test",
        "truth_status": "frozen_unscored",
        "allow_final_claim": True,
        "construction_model_blind": True,
        "performance_scores_opened": False,
        "leakage_guard": {
            "exclusion_policy_complete": True,
            "exclusion_registry_sha256": audit_report["exclusion_registry_sha256"],
            "audit_report_sha256": sha256(args.audit / "report.json"),
            "audit_manifest_sha256": audit_report["manifest_sha256"],
            "source_mgf_sha256": audit_report["source_mgf_sha256"],
            "selected_consumed_identity_overlap": int(selected.consumed_identity_overlap.astype(bool).sum()),
            "selected_consumed_formula_overlap_identity_panel_allowed": int(selected.consumed_formula_overlap.astype(bool).sum()),
            "selected_consumed_spectrum_overlap": int(selected.consumed_spectrum_overlap.astype(bool).sum()),
        },
        "query_condition": "maximum collision energy",
        "reference_condition": "different, lower collision energy",
        "candidate_window_ppm": args.ppm,
        "maximum_queries_per_panel": args.max_queries,
        "model_blind_sampling_seed": args.sampling_seed,
        "open_set_match_fraction": args.open_set_match_fraction,
        "identity_disjoint": panel_summary(identity),
        "formula_disjoint": panel_summary(formula),
        "identity_open_set": panel_summary(identity_open),
        "formula_open_set": panel_summary(formula_open),
        "curated_spectra": len(selected),
        "checksums": checksums,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
