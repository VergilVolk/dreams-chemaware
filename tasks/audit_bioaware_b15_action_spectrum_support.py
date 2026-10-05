#!/usr/bin/env python
"""Materialise and audit spectrum support for the frozen B12 actions.

B12 is an opened, six-domain candidate-routing experiment.  This stage asks a
strictly narrower question: can each B12 query and each truth/baseline/proposed
candidate be traced to an actual spectrum that a shared DreaMS encoder can
consume?  It does not fit a model and it does not use P2b.

KGMN hidden-seed repeats reuse the same physical MSP spectrum.  They remain
separate evaluation rows in the B12 replay, but are explicitly collapsed to a
``physical_query_id`` for training-support counts so repeated seed masks cannot
inflate gradient coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INTERNAL_UNITS = (
    "BV2cell__hilic", "BV2cell__rplc", "Mouse_brain__hilic",
    "Mouse_brain__rplc", "Mouse_liver__hilic", "Mouse_liver__rplc",
    "NIST_plasma__hilic", "NIST_plasma__rplc",
)
INTERNAL_SOURCES = {"BV2cell", "Mouse_brain", "Mouse_liver", "NIST_plasma"}
ST_SOURCE = "ST001154_same_formula_10ppm"
KGM_SOURCE = "KGMN200STD_hidden_seed"


def sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def fixed_tensor(raw: np.ndarray, precursor_mz: float, n_peaks: int = 100) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] != 2 or raw.shape[1] < 1:
        raise RuntimeError(f"invalid raw spectrum shape: {raw.shape}")
    valid = np.isfinite(raw).all(axis=0) & (raw[0] > 0) & (raw[1] > 0)
    raw = raw[:, valid]
    if raw.shape[1] < 1 or not np.isfinite(precursor_mz) or precursor_mz <= 0:
        raise RuntimeError("empty spectrum or invalid precursor")
    highest = np.argsort(raw[1], kind="stable")[-n_peaks:]
    highest = np.sort(highest)
    peaks = raw[:, highest].T.astype(np.float32, copy=True)
    if len(peaks) < n_peaks:
        peaks = np.pad(peaks, ((0, n_peaks - len(peaks)), (0, 0)))
    maximum = float(peaks[:, 1].max())
    if maximum > 0:
        peaks[:, 1] /= maximum
    return np.vstack((
        np.asarray([[precursor_mz, 1.1]], dtype=np.float32), peaks,
    )).astype(np.float32)


def load_msp_query_tensors(path: Path, wanted: set[str]) -> dict[str, np.ndarray]:
    """Parse the small KGMN MSP without importing the optional DreaMS IO stack."""
    output: dict[str, np.ndarray] = {}
    name: str | None = None
    precursor: float | None = None
    expected_peaks: int | None = None
    peaks: list[tuple[float, float]] = []

    def finish() -> None:
        nonlocal name, precursor, expected_peaks, peaks
        if name is None:
            return
        if expected_peaks is None or len(peaks) != expected_peaks:
            raise RuntimeError(
                f"MSP peak count mismatch for {name}: {len(peaks)} != {expected_peaks}"
            )
        if name in wanted:
            if name in output or precursor is None:
                raise RuntimeError(f"duplicate or precursor-free MSP record: {name}")
            output[name] = fixed_tensor(
                np.asarray(peaks, dtype=np.float32).T, float(precursor), 100
            )
        name, precursor, expected_peaks, peaks = None, None, None, []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                finish()
                continue
            if line.casefold().startswith("name:"):
                finish()
                name = line.split(":", 1)[1].strip()
            elif line.casefold().startswith("precursormz:"):
                precursor = float(line.split(":", 1)[1].strip())
            elif line.casefold().startswith("num peaks:"):
                expected_peaks = int(line.split(":", 1)[1].strip())
            elif name is not None and expected_peaks is not None:
                fields = line.split()
                if len(fields) >= 2:
                    peaks.append((float(fields[0]), float(fields[1])))
    finish()
    missing = wanted - set(output)
    if missing:
        raise RuntimeError(f"KGMN MSP misses {len(missing)} queries: {sorted(missing)[:5]}")
    return output


def load_mona_reference_tensors(
    mgf_path: Path,
    manifest_path: Path,
    selected_rows: np.ndarray,
) -> np.ndarray:
    library = pd.read_csv(manifest_path)
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    if (
        selected_rows.ndim != 1 or len(selected_rows) == 0
        or selected_rows[0] < 0 or selected_rows[-1] >= len(library)
        or np.any(np.diff(selected_rows) <= 0)
    ):
        raise RuntimeError("selected MoNA rows are not a sorted unique in-range vector")
    wanted = set(map(int, selected_rows))
    tensors: dict[int, np.ndarray] = {}
    record_index = -1
    current: dict[str, str | float] | None = None
    peaks: list[tuple[float, float]] = []

    def finish_record() -> None:
        nonlocal record_index
        if current is None or not peaks or not current.get("precursor_mz"):
            return
        record_index += 1
        if record_index not in wanted:
            return
        precursor = float(current["precursor_mz"])
        expected = library.iloc[record_index]
        if not np.isclose(precursor, float(expected["precursor_mz"]), rtol=0, atol=1e-6):
            raise RuntimeError(f"MoNA precursor mismatch at row {record_index}")
        if str(current.get("inchikey", "")).upper() != str(expected["inchikey"]).upper():
            raise RuntimeError(f"MoNA identity mismatch at row {record_index}")
        tensors[record_index] = fixed_tensor(
            np.asarray(peaks, dtype=np.float32).T, precursor, 100
        )

    with mgf_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                current, peaks = {}, []
            elif line == "END IONS":
                finish_record()
                current, peaks = None, []
            elif current is not None and "=" in line:
                key, value = line.split("=", 1)
                if key == "PEPMASS":
                    current["precursor_mz"] = float(value.strip().split()[0])
                elif key == "INCHIKEY":
                    current["inchikey"] = value.strip()
            elif current is not None:
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        peaks.append((float(fields[0]), float(fields[1])))
                    except ValueError:
                        pass
    if current is not None:
        raise RuntimeError("unterminated MoNA MGF record")
    if record_index + 1 != len(library):
        raise RuntimeError(
            f"MoNA MGF/manifest count mismatch: {record_index + 1} != {len(library)}"
        )
    missing = wanted - set(tensors)
    if missing:
        raise RuntimeError(f"MoNA MGF misses {len(missing)} selected rows")
    return np.stack([tensors[int(row)] for row in selected_rows]).astype(np.float32)


def normalise_query_id(source: str, query_id: str) -> tuple[str, str]:
    if source == ST_SOURCE:
        prefix = ST_SOURCE + "::"
        if not query_id.startswith(prefix):
            raise RuntimeError(f"ST query lacks frozen domain prefix: {query_id}")
        raw = query_id[len(prefix):]
        return raw, raw
    if source == KGM_SOURCE:
        prefix = KGM_SOURCE + "::"
        if not query_id.startswith(prefix) or "::repeat=" not in query_id:
            raise RuntimeError(f"KGMN query lacks frozen repeat encoding: {query_id}")
        raw = query_id[len(prefix):].split("::repeat=", 1)[0]
        return raw, KGM_SOURCE + "::" + raw
    if source not in INTERNAL_SOURCES:
        raise RuntimeError(f"unexpected B12 source: {source}")
    return query_id, query_id


def candidate_reference_map(args: argparse.Namespace) -> dict[tuple[str, str, str], int]:
    mapping: dict[tuple[str, str, str], int] = {}

    internal = pd.read_csv(args.internal_candidates)
    internal = internal.loc[internal["polarity"].eq("negative")].copy()
    for row in internal.itertuples(index=False):
        key = (str(row.source), str(row.query_id), str(row.candidate_id))
        value = int(row.best_library_row)
        if key in mapping and mapping[key] != value:
            raise RuntimeError(f"ambiguous internal best reference: {key}")
        mapping[key] = value

    st = pd.read_csv(args.st_manifest_dir / "candidate_references.csv.gz")
    st_eval = pd.read_csv(args.st_candidate_scores)
    st_queries = pd.read_csv(args.st_manifest_dir / "queries.csv.gz")
    st_query_embeddings = np.load(
        args.st_evaluation_dir / "query_embeddings.npy", allow_pickle=False
    ).astype(np.float32)
    evaluated_query_ids = set(st_eval["query_id"].astype(str))
    st_queries = st_queries.loc[
        st_queries["query_id"].astype(str).isin(evaluated_query_ids)
    ].reset_index(drop=True)
    if len(st_queries) != len(st_query_embeddings):
        raise RuntimeError("ST query embedding order cannot be reconstructed")
    query_embedding = dict(zip(
        st_queries["query_id"].astype(str), st_query_embeddings, strict=True
    ))
    mona_embeddings = np.load(args.mona_embeddings, mmap_mode="r")
    for (qid, candidate), group in st.groupby(["query_id", "candidate_id"], sort=False):
        qid, candidate = str(qid), str(candidate)
        if qid not in evaluated_query_ids:
            continue
        expected = st_eval.loc[
            st_eval["query_id"].astype(str).eq(qid)
            & st_eval["candidate_id"].astype(str).eq(candidate),
            "spectral_score",
        ]
        # The manifest contains the original mass-window graph.  B12 used the
        # chemically stricter same-formula graph emitted by the evaluation.
        if expected.empty:
            continue
        rows = group["library_row"].to_numpy(np.int64)
        scores = np.asarray(mona_embeddings[rows], dtype=np.float32) @ query_embedding[qid]
        best = int(np.argmax(scores))
        value = int(rows[best])
        if len(expected) != 1 or not np.isclose(
            float(scores[best]), float(expected.iloc[0]), rtol=0, atol=2e-6
        ):
            raise RuntimeError(f"ST exact best-reference replay failed: {qid}/{candidate}")
        mapping[(ST_SOURCE, qid, candidate)] = value

    kgm = pd.read_csv(args.kgmn_manifest_dir / "candidate_scores.csv.gz")
    for row in kgm.itertuples(index=False):
        key = (KGM_SOURCE, str(row.query_id), str(row.candidate_id))
        value = int(row.best_library_row)
        if key in mapping and mapping[key] != value:
            raise RuntimeError(f"ambiguous KGMN best reference: {key}")
        mapping[key] = value
    return mapping


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b12-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b12_multicohort_catalog_localcheck_20260907_v1",
    )
    parser.add_argument(
        "--internal-unit-dir", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_units_v2",
    )
    parser.add_argument(
        "--internal-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-manifest-dir", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_manifest_v1",
    )
    parser.add_argument(
        "--st-candidate-scores", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-evaluation-dir", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1",
    )
    parser.add_argument(
        "--kgmn-manifest-dir", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2",
    )
    parser.add_argument(
        "--kgmn-msp", type=Path,
        default=ROOT / "third_party/MetDNA2/inst/extdata/spectra_200STD_neg_200805.msp",
    )
    parser.add_argument(
        "--mona-mgf", type=Path, default=ROOT / "data/models/mona_neg_full.mgf",
    )
    parser.add_argument(
        "--mona-manifest", type=Path,
        default=ROOT / "data/models/mona_neg_dreams_emb/manifest.csv",
    )
    parser.add_argument(
        "--mona-embeddings", type=Path,
        default=ROOT / "data/models/mona_neg_dreams_emb/embeddings.npy",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    required = [
        args.b12_dir / "report.json",
        args.b12_dir / "nested_domain_loso_transitions.csv.gz",
        args.internal_candidates,
        args.st_manifest_dir / "queries.csv.gz",
        args.st_manifest_dir / "query_tensors.npz",
        args.st_manifest_dir / "candidate_references.csv.gz",
        args.st_candidate_scores,
        args.st_evaluation_dir / "query_embeddings.npy",
        args.kgmn_manifest_dir / "candidate_scores.csv.gz",
        args.kgmn_msp,
        args.mona_mgf,
        args.mona_manifest,
        args.mona_embeddings,
    ]
    for unit in INTERNAL_UNITS:
        required.extend([
            args.internal_unit_dir / unit / "queries.csv.gz",
            args.internal_unit_dir / unit / "query_tensors.npz",
        ])
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    b12_report = json.loads((args.b12_dir / "report.json").read_text(encoding="utf-8"))
    if b12_report.get("pass_to_shared_embedding_action_construction") is not True:
        raise RuntimeError("B12 did not pass its frozen action-discovery gate")
    transitions = pd.read_csv(args.b12_dir / "nested_domain_loso_transitions.csv.gz")
    if (
        len(transitions) != 860 or transitions["query_id"].nunique() != 860
        or int(transitions["corrected"].sum()) != 58
        or int(transitions["introduced"].sum()) != 8
    ):
        raise RuntimeError("B12 replay counts changed")

    tensor_by_source_query: dict[tuple[str, str], np.ndarray] = {}
    for unit in INTERNAL_UNITS:
        queries = pd.read_csv(args.internal_unit_dir / unit / "queries.csv.gz")
        tensors = np.load(
            args.internal_unit_dir / unit / "query_tensors.npz", allow_pickle=False
        )["query_tensor"]
        if len(queries) != len(tensors) or queries["query_id"].duplicated().any():
            raise RuntimeError(f"unaligned internal unit tensors: {unit}")
        source = unit.split("__", 1)[0]
        for qid, tensor in zip(queries["query_id"].astype(str), tensors, strict=True):
            tensor_by_source_query[(source, qid)] = np.asarray(tensor, dtype=np.float32)

    st_queries = pd.read_csv(args.st_manifest_dir / "queries.csv.gz")
    st_tensors = np.load(
        args.st_manifest_dir / "query_tensors.npz", allow_pickle=False
    )["query_tensor"]
    if len(st_queries) != len(st_tensors) or st_queries["query_id"].duplicated().any():
        raise RuntimeError("unaligned ST001154 tensors")
    for row, tensor in zip(st_queries.itertuples(index=False), st_tensors, strict=True):
        tensor_by_source_query[(ST_SOURCE, str(row.query_id))] = np.asarray(
            tensor, dtype=np.float32
        )

    normalised = [
        normalise_query_id(str(row.source), str(row.query_id))
        for row in transitions.itertuples(index=False)
    ]
    transitions["source_query_id"] = [item[0] for item in normalised]
    transitions["physical_query_id"] = [item[1] for item in normalised]
    kgm_names = set(
        transitions.loc[transitions["source"].eq(KGM_SOURCE), "source_query_id"].astype(str)
    )
    for qid, tensor in load_msp_query_tensors(args.kgmn_msp, kgm_names).items():
        tensor_by_source_query[(KGM_SOURCE, qid)] = tensor

    query_tensors: list[np.ndarray] = []
    missing_query: list[str] = []
    for row in transitions.itertuples(index=False):
        key = (str(row.source), str(row.source_query_id))
        tensor = tensor_by_source_query.get(key)
        if tensor is None:
            missing_query.append(str(row.query_id))
            continue
        if tensor.shape != (101, 2) or not np.isfinite(tensor).all():
            raise RuntimeError(f"invalid query tensor for {row.query_id}: {tensor.shape}")
        query_tensors.append(tensor)
    if missing_query:
        raise RuntimeError(f"missing {len(missing_query)} B12 query spectra")

    reference_map = candidate_reference_map(args)
    reference_columns: dict[str, list[int]] = {
        "truth": [], "baseline": [], "proposed": [],
    }
    missing_reference: list[tuple[str, str, str]] = []
    for row in transitions.itertuples(index=False):
        for role, candidate in (
            ("truth", row.truth_candidate_id),
            ("baseline", row.baseline_candidate_id),
            ("proposed", row.proposed_candidate_id),
        ):
            key = (str(row.source), str(row.source_query_id), str(candidate))
            value = reference_map.get(key)
            if value is None:
                missing_reference.append(key)
            else:
                reference_columns[role].append(int(value))
    if missing_reference:
        raise RuntimeError(
            f"missing {len(missing_reference)} query/candidate reference rows: "
            f"{missing_reference[:5]}"
        )

    selected_reference_rows = np.asarray(sorted(set(
        reference_columns["truth"]
        + reference_columns["baseline"]
        + reference_columns["proposed"]
    )), dtype=np.int64)
    reference_tensors = load_mona_reference_tensors(
        args.mona_mgf, args.mona_manifest, selected_reference_rows
    )
    row_position = {
        int(row): int(position) for position, row in enumerate(selected_reference_rows)
    }
    for role, values in reference_columns.items():
        transitions[f"{role}_reference_row"] = values
        transitions[f"{role}_reference_position"] = [row_position[value] for value in values]

    transitions["physical_duplicate_weight"] = 1.0 / transitions.groupby(
        ["source", "physical_query_id"]
    )["query_id"].transform("size").astype(float)
    transitions["identity_equal_weight"] = 1.0 / transitions.groupby(
        "truth_candidate_id"
    )["query_id"].transform("size").astype(float)

    corrected = transitions.loc[transitions["corrected"]].copy()
    introduced = transitions.loc[transitions["introduced"]].copy()
    physical_corrected = corrected.drop_duplicates(["source", "physical_query_id"])
    physical_introduced = introduced.drop_duplicates(["source", "physical_query_id"])
    per_domain: dict[str, dict] = {}
    for source, frame in transitions.groupby("source", sort=True):
        local_corrected = frame.loc[frame["corrected"]]
        local_introduced = frame.loc[frame["introduced"]]
        per_domain[str(source)] = {
            "evaluation_rows": int(len(frame)),
            "physical_query_spectra": int(frame["physical_query_id"].nunique()),
            "corrected_rows": int(len(local_corrected)),
            "corrected_physical_queries": int(local_corrected["physical_query_id"].nunique()),
            "corrected_identities": int(local_corrected["truth_candidate_id"].nunique()),
            "introduced_rows": int(len(local_introduced)),
            "introduced_physical_queries": int(local_introduced["physical_query_id"].nunique()),
        }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    transition_path = args.output_dir / "spectrum_supported_actions.csv.gz"
    transitions.to_csv(transition_path, index=False, compression="gzip")
    manifest_path = args.output_dir / "spectrum_support_manifest.npz"
    np.savez_compressed(
        manifest_path,
        query_id=transitions["query_id"].to_numpy(dtype=str),
        source=transitions["source"].to_numpy(dtype=str),
        source_query_id=transitions["source_query_id"].to_numpy(dtype=str),
        physical_query_id=transitions["physical_query_id"].to_numpy(dtype=str),
        truth_candidate_id=transitions["truth_candidate_id"].to_numpy(dtype=str),
        truth_formula=transitions["truth_formula"].to_numpy(dtype=str),
        query_tensor=np.stack(query_tensors).astype(np.float32),
        truth_reference_position=transitions["truth_reference_position"].to_numpy(np.int64),
        baseline_reference_position=transitions["baseline_reference_position"].to_numpy(np.int64),
        proposed_reference_position=transitions["proposed_reference_position"].to_numpy(np.int64),
        reference_tensor_rows=selected_reference_rows,
        reference_tensor=reference_tensors,
        corrected=transitions["corrected"].to_numpy(bool),
        introduced=transitions["introduced"].to_numpy(bool),
        physical_duplicate_weight=transitions["physical_duplicate_weight"].to_numpy(np.float32),
        identity_equal_weight=transitions["identity_equal_weight"].to_numpy(np.float32),
    )

    gates = {
        "all_860_evaluation_rows_have_query_spectra": len(query_tensors) == 860,
        "all_truth_baseline_proposed_candidates_have_reference_spectra": (
            len(missing_reference) == 0
        ),
        "corrective_physical_queries_ge_50": len(physical_corrected) >= 50,
        "corrective_identities_ge_25": corrected["truth_candidate_id"].nunique() >= 25,
        "corrective_formulas_ge_25": corrected["truth_formula"].nunique() >= 25,
        "introduced_physical_queries_le_8": len(physical_introduced) <= 8,
    }
    report = {
        "status": "bioaware_b15_action_spectrum_support_complete",
        "formal": True,
        "B12_replay": {
            "evaluation_rows": int(len(transitions)),
            "physical_query_spectra": int(transitions["physical_query_id"].nunique()),
            "truth_identities": int(transitions["truth_candidate_id"].nunique()),
            "truth_formulas": int(transitions["truth_formula"].nunique()),
            "corrected_rows": int(len(corrected)),
            "corrected_physical_queries": int(len(physical_corrected)),
            "corrected_identities": int(corrected["truth_candidate_id"].nunique()),
            "corrected_formulas": int(corrected["truth_formula"].nunique()),
            "introduced_rows": int(len(introduced)),
            "introduced_physical_queries": int(len(physical_introduced)),
            "introduced_identities": int(introduced["truth_candidate_id"].nunique()),
            "introduced_formulas": int(introduced["truth_formula"].nunique()),
        },
        "per_domain": per_domain,
        "materialisation": {
            "query_tensors": int(len(query_tensors)),
            "unique_candidate_reference_spectra": int(len(selected_reference_rows)),
            "query_tensor_shape": list(np.stack(query_tensors).shape),
            "reference_tensor_shape": list(reference_tensors.shape),
        },
        "gates": gates,
        "pass_to_direct_shared_embedding_design": bool(all(gates.values())),
        "contracts": {
            "model_fitted": False,
            "B12_outcomes_changed": False,
            "KGMN_seed_repeats_counted_as_independent_spectra": False,
            "identity_equal_weights_exported": True,
            "reaction_neighbours_used_as_positive_identity": False,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "provenance": {
            "B12_report_sha256": sha256(args.b12_dir / "report.json"),
            "B12_transitions_sha256": sha256(
                args.b12_dir / "nested_domain_loso_transitions.csv.gz"
            ),
            "internal_candidates_sha256": sha256(args.internal_candidates),
            "st_queries_sha256": sha256(args.st_manifest_dir / "queries.csv.gz"),
            "st_query_tensors_sha256": sha256(
                args.st_manifest_dir / "query_tensors.npz"
            ),
            "kgmn_msp_sha256": sha256(args.kgmn_msp),
            "mona_mgf_sha256": sha256(args.mona_mgf),
            "mona_manifest_sha256": sha256(args.mona_manifest),
            "mona_embeddings_sha256": sha256(args.mona_embeddings),
            "transitions_sha256": sha256(transition_path),
            "manifest_sha256": sha256(manifest_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "B15 establishes exact spectrum materialisability and independent-gradient "
            "support for the opened B12 action. It is not a trained embedding, a blind "
            "performance result, or evidence that a shared encoder will realize +5.81 pp."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["pass_to_direct_shared_embedding_design"]:
        raise RuntimeError(f"B15 spectrum-support gate failed: {gates}")


if __name__ == "__main__":
    main()
