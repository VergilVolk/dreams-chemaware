"""One-shot evaluation of the frozen P2b rank fusion on sealed P3.

The script is intentionally fail-closed.  It verifies the complete artifact,
checkpoint, P3 manifest, reference-library and candidate-graph hash chain
before computing any score.  No parameter is fit or selected here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from audit_large_observability_residual import symmetric_features  # noqa: E402
from build_g8r_p2_listwise_cache import load_or_build_embeddings, numeric_values, text_values  # noqa: E402
from g8r_p2_rank_fusion_core import (  # noqa: E402
    FusionConfiguration,
    fusion_configuration_from_mapping,
    fuse_one_query,
    grouped_max,
    normalize_pair_features,
    strict_rank,
)


DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_PAIRS = ROOT / "tasks/massspecgym_isomers/pairs.json"
DEFAULT_P3 = ROOT / "data/validation/g8r_p3_test"
DEFAULT_ARTIFACT = ROOT / "data/validation/g8r_p2b_rank_fusion.json"
DEFAULT_SELECTION = ROOT / "data/validation/g8r_p2b_rank_fusion.selection.json"
DEFAULT_P2_AUDIT = ROOT / "data/validation/g8r_p2_listwise_cache.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_EMBED = ROOT / "data/validation/g8r_p3_official_embeddings.npz"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_p2b_p3_final.json"


PANEL_FILES = {
    "P3-main-real-pristine": "p3_main_real_pristine_manifest.json",
    "P3-isomer-real-pristine": "p3_isomer_real_pristine_manifest.json",
    "P3-near-core-real-pristine": "p3_near_core_real_pristine_manifest.json",
    "P3-nearmid-real-pristine": "p3_nearmid_real_pristine_manifest.json",
    "P3-isomer-real-exposed-extension": "p3_isomer_real_exposed_extension_manifest.json",
    "P3-sim-to-real-secondary": "p3_sim_to_real_secondary_manifest.json",
}
METHODS = ("dreams", "sqrt_cosine", "entropy", "neutral_loss", "p2b_frozen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--p3-dir", type=Path, default=DEFAULT_P3)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--p2-cache-audit", type=Path, default=DEFAULT_P2_AUDIT)
    parser.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--peak-tolerance", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(body) -> str:
    payload = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_hash(body: dict) -> str:
    copy = dict(body)
    expected = str(copy.pop("query_manifest_sha256"))
    actual = sha256_json(copy)
    if actual != expected:
        raise RuntimeError(f"query manifest self-hash mismatch: {actual} != {expected}")
    return expected


def exact_mcnemar(corrected: int, introduced: int) -> float:
    discordant = corrected + introduced
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(corrected, introduced) + 1))
    return float(min(1.0, 2.0 * tail / (2 ** discordant)))


def macro_auc(scores: np.ndarray, positive_index: int = 0) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[positive_index]
    negatives = np.delete(scores, positive_index)
    return float(np.mean((positive > negatives).astype(float) + 0.5 * (positive == negatives)))


def paired_cluster_bootstrap(
    formulas: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    formulas = np.asarray(formulas, dtype=object)
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    groups = {
        formula: difference[formulas == formula]
        for formula in sorted(set(map(str, formulas)))
    }
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(names), len(names))
        values = np.concatenate([groups[names[index]] for index in sampled])
        draws[iteration] = values.mean()
    return {
        "mean_delta": float(difference.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def validate_provenance(args: argparse.Namespace):
    required = [
        args.data, args.pairs, args.artifact, args.selection, args.p2_cache_audit,
        args.base_ckpt, args.architecture_ckpt,
        args.p3_dir / "p3_lock_summary.json",
        args.p3_dir / "p3_evaluation_protocol.json",
        args.p3_dir / "p3_reference_library_real.json",
        args.p3_dir / "p3_p2_allowed_training_ik14.json",
        ROOT / "tasks/train_g8r_p2b_rank_fusion.py",
        ROOT / "tasks/build_g8r_p3_test.py",
    ] + [args.p3_dir / filename for filename in PANEL_FILES.values()]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    p2_audit = json.loads(args.p2_cache_audit.read_text(encoding="utf-8"))
    summary = json.loads((args.p3_dir / "p3_lock_summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.p3_dir / "p3_evaluation_protocol.json").read_text(encoding="utf-8"))
    library = json.loads((args.p3_dir / "p3_reference_library_real.json").read_text(encoding="utf-8"))
    allow = json.loads((args.p3_dir / "p3_p2_allowed_training_ik14.json").read_text(encoding="utf-8"))

    if artifact.get("status") != "g8r_p2b_rank_fusion_frozen":
        raise RuntimeError("P2b artifact is not frozen")
    if artifact.get("p3_used_for_training_or_selection") is not False:
        raise RuntimeError("P2b artifact does not exclude P3")
    if selection.get("status") != "g8r_p2b_selection_passed" or not selection.get("gates", {}).get("pass"):
        raise RuntimeError("P2b selection did not pass")
    if artifact.get("selection_report_sha256") != sha256_file(args.selection):
        raise RuntimeError("P2b selection hash mismatch")
    if artifact.get("selection_script_sha256") != sha256_file(ROOT / "tasks/train_g8r_p2b_rank_fusion.py"):
        raise RuntimeError("P2b training script drifted after freezing")
    if artifact.get("cache_audit_sha256") != sha256_file(args.p2_cache_audit):
        raise RuntimeError("P2 cache-audit hash mismatch")
    if p2_audit.get("base_checkpoint_sha256") != sha256_file(args.base_ckpt):
        raise RuntimeError("official DreaMS checkpoint mismatch")
    if summary.get("status") != "g8r_p3_v3_sealed":
        raise RuntimeError("P3 is not formally sealed")
    if summary.get("build_script_sha256") != sha256_file(ROOT / "tasks/build_g8r_p3_test.py"):
        raise RuntimeError("P3 build script drifted after sealing")
    if summary.get("hdf5_sha256") != sha256_file(args.data):
        raise RuntimeError("P3 HDF5 hash mismatch")
    if summary.get("pairs_json_sha256") != sha256_file(args.pairs):
        raise RuntimeError("P3 pairs.json hash mismatch")
    if summary.get("evaluation_protocol_sha256") != sha256_json(protocol):
        raise RuntimeError("P3 protocol hash mismatch")
    if summary.get("reference_library_sha256") != sha256_json(library):
        raise RuntimeError("P3 reference-library hash mismatch")

    manifests = {}
    for panel, filename in PANEL_FILES.items():
        body = json.loads((args.p3_dir / filename).read_text(encoding="utf-8"))
        if body.get("panel") != panel:
            raise RuntimeError(f"panel label mismatch in {filename}")
        digest = manifest_hash(body)
        if summary.get("manifest_sha256", {}).get(panel) != digest:
            raise RuntimeError(f"P3 summary/manifest hash mismatch for {panel}")
        if body.get("protocol_sha256") != summary.get("evaluation_protocol_sha256"):
            raise RuntimeError(f"protocol mismatch for {panel}")
        if body.get("candidate_library_sha256") != summary.get("reference_library_sha256"):
            raise RuntimeError(f"reference-library mismatch for {panel}")
        graph = [{"row": q["row"], "candidate_rows": q["candidate_rows"]} for q in body["queries"]]
        if body.get("candidate_graph_sha256") != sha256_json(graph):
            raise RuntimeError(f"candidate graph mismatch for {panel}")
        manifests[panel] = body

    p2_ik = set(map(str, allow["real_train_primary"]["ik14"]))
    p3_ik = {str(query["ik14"]) for body in manifests.values() for query in body["queries"]}
    if p2_ik & p3_ik:
        raise RuntimeError("P2/P3 identity overlap detected at evaluation time")
    return artifact, summary, protocol, library, manifests


def validate_candidate_graphs(
    manifests: dict,
    library: dict,
    row_metadata: dict[int, tuple[str, str, float]],
    ppm: float,
) -> None:
    library_rows = np.asarray(library["rows"], dtype=np.int64)
    library_set = set(map(int, library_rows))
    adduct_groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for adduct in sorted({row_metadata[int(row)][1] for row in library_rows}):
        rows = np.asarray([int(row) for row in library_rows if row_metadata[int(row)][1] == adduct])
        masses = np.asarray([row_metadata[int(row)][2] for row in rows])
        order = np.argsort(masses, kind="mergesort")
        adduct_groups[adduct] = rows[order], masses[order]

    for panel, body in manifests.items():
        for query in body["queries"]:
            qrow = int(query["row"])
            qik, qadduct, qmass = row_metadata[qrow]
            rows = list(map(int, query["candidate_rows"]))
            if len(rows) != len(set(rows)) or qrow in rows or not set(rows) <= library_set:
                raise RuntimeError(f"invalid candidate rows for {panel}/{qrow}")
            group, masses = adduct_groups[qadduct]
            tolerance = ppm * 1e-6 * qmass
            left = np.searchsorted(masses, qmass - tolerance, side="left")
            right = np.searchsorted(masses, qmass + tolerance, side="right")
            expected = set(map(int, group[left:right])) - {qrow}
            if set(rows) != expected:
                raise RuntimeError(f"candidate protocol mismatch for {panel}/{qrow}")
            identities = {row_metadata[row][0] for row in rows}
            if qik not in identities or not (identities - {qik}):
                raise RuntimeError(f"query lacks a positive or negative molecule: {panel}/{qrow}")


def evaluate_panel(
    panel: str,
    manifest: dict,
    row_to_index: dict[int, int],
    embeddings: np.ndarray,
    spectra: np.ndarray,
    precursor: np.ndarray,
    ik14: np.ndarray,
    formula_by_row: dict[int, str],
    configuration: FusionConfiguration,
    peak_tolerance: float,
    pair_cache: dict[tuple[int, int], dict[str, float]],
    bootstrap: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    records = []
    for position, query in enumerate(manifest["queries"], start=1):
        qrow = int(query["row"])
        qindex = row_to_index[qrow]
        qik = str(ik14[qindex])
        grouped: dict[str, list[int]] = defaultdict(list)
        for candidate_row in map(int, query["candidate_rows"]):
            grouped[str(ik14[row_to_index[candidate_row]])].append(candidate_row)
        identities = [qik] + sorted(identity for identity in grouped if identity != qik)
        pair_features = []
        molecule_ptr = [0]
        for identity in identities:
            for candidate_row in sorted(grouped[identity]):
                cindex = row_to_index[candidate_row]
                dreams = float(embeddings[qindex] @ embeddings[cindex])
                key = (min(qrow, candidate_row), max(qrow, candidate_row))
                raw = pair_cache.get(key)
                if raw is None:
                    raw = symmetric_features(
                        spectra[qindex], float(precursor[qindex]),
                        spectra[cindex], float(precursor[cindex]), peak_tolerance,
                    )
                    pair_cache[key] = raw
                pair_features.append([
                    dreams,
                    float(raw["sqrt_cosine"]),
                    float(raw["entropy_similarity"]),
                    float(raw["neutral_loss_sqrt_cosine"]),
                ])
            molecule_ptr.append(len(pair_features))
        values = np.asarray(pair_features, dtype=np.float64)
        ptr = np.asarray(molecule_ptr, dtype=np.int64)
        normalized = normalize_pair_features(values, np.asarray([0, len(values)]), configuration.normalization)
        p2b_scores, intervened, support = fuse_one_query(
            normalized, values[:, 0], ptr, np.asarray(configuration.weights),
            (1, 2, 3), configuration.min_support, configuration.min_advantage,
        )
        molecule_scores = {
            "dreams": grouped_max(values[:, 0], ptr),
            "sqrt_cosine": grouped_max(normalized[:, 1], ptr),
            "entropy": grouped_max(normalized[:, 2], ptr),
            "neutral_loss": grouped_max(normalized[:, 3], ptr),
            "p2b_frozen": p2b_scores,
        }
        row = {
            "panel": panel,
            "query_row": qrow,
            "ik14": qik,
            "formula": formula_by_row[qrow],
            "n_candidate_molecules": len(identities),
            "p2b_intervened": bool(intervened),
            "p2b_support": int(support),
        }
        for method, scores in molecule_scores.items():
            rank, mrr, margin = strict_rank(scores, 0)
            row[f"{method}_rank"] = rank
            row[f"{method}_top1"] = rank == 1
            row[f"{method}_mrr"] = mrr
            row[f"{method}_margin"] = margin
            row[f"{method}_auc"] = macro_auc(scores, 0)
            row[f"{method}_best_negative_ik14"] = identities[1 + int(np.argmax(scores[1:]))]
        records.append(row)
        if position % 100 == 0 or position == len(manifest["queries"]):
            print(f"[{panel}] {position:,}/{len(manifest['queries']):,} queries", flush=True)

    formulas = np.asarray([record["formula"] for record in records], dtype=object)
    summary = {"n_queries": len(records), "n_formulas": len(set(formulas))}
    for method in METHODS:
        summary[method] = {
            "recall1": float(np.mean([record[f"{method}_top1"] for record in records])),
            "mrr": float(np.mean([record[f"{method}_mrr"] for record in records])),
            "macro_query_auc": float(np.mean([record[f"{method}_auc"] for record in records])),
        }
    baseline_top1 = np.asarray([record["dreams_top1"] for record in records], dtype=bool)
    final_top1 = np.asarray([record["p2b_frozen_top1"] for record in records], dtype=bool)
    corrected = int(np.sum((~baseline_top1) & final_top1))
    introduced = int(np.sum(baseline_top1 & (~final_top1)))
    summary["p2b_vs_dreams"] = {
        "corrected": corrected,
        "introduced": introduced,
        "mcnemar_exact_p": exact_mcnemar(corrected, introduced),
        "intervention_rate": float(np.mean([record["p2b_intervened"] for record in records])),
    }
    for metric in ("top1", "mrr", "auc"):
        summary["p2b_vs_dreams"][f"{metric}_paired_formula_bootstrap"] = paired_cluster_bootstrap(
            formulas,
            np.asarray([record[f"p2b_frozen_{metric}"] for record in records]),
            np.asarray([record[f"dreams_{metric}"] for record in records]),
            bootstrap,
            seed + {"top1": 0, "mrr": 1, "auc": 2}[metric],
        )
    summary["p2b_vs_neutral_loss"] = {}
    for metric in ("top1", "mrr", "auc"):
        summary["p2b_vs_neutral_loss"][f"{metric}_paired_formula_bootstrap"] = paired_cluster_bootstrap(
            formulas,
            np.asarray([record[f"p2b_frozen_{metric}"] for record in records]),
            np.asarray([record[f"neutral_loss_{metric}"] for record in records]),
            bootstrap,
            seed + 10 + {"top1": 0, "mrr": 1, "auc": 2}[metric],
        )
    return summary, records


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_suffix(".per_query.csv").exists():
        raise FileExistsError("P3 result already exists; one-shot evaluation refuses to overwrite")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    artifact, lock, protocol, library, manifests = validate_provenance(args)
    configuration = fusion_configuration_from_mapping(artifact["configuration"])
    if tuple(artifact.get("selected_features", [])) != (
        "dreams_similarity", "sqrt_cosine", "entropy_similarity", "neutral_loss_sqrt_cosine",
    ):
        raise RuntimeError("unsupported frozen P2b feature schema")
    ppm = float(protocol["candidate_filter"]["precursor_ppm"])
    if protocol["candidate_filter"].get("same_adduct") is not True:
        raise RuntimeError("P3 protocol is not same-adduct")

    all_rows = sorted({
        int(row)
        for body in manifests.values()
        for query in body["queries"]
        for row in [query["row"], *query["candidate_rows"]]
    })
    rows = np.asarray(all_rows, dtype=np.int64)
    with h5py.File(args.data, "r") as handle:
        precursor = numeric_values(handle["precursor_mz"], rows).astype(np.float64)
        inchikey = text_values(handle["INCHIKEY"], rows)
        ik14 = np.asarray([value[:14] for value in inchikey], dtype=object)
        formulas = text_values(handle["FORMULA"], rows)
        adducts = text_values(handle["adduct"], rows)
    row_to_index = {int(row): index for index, row in enumerate(rows)}
    row_metadata = {
        int(row): (str(ik14[index]), str(adducts[index]), float(precursor[index]))
        for index, row in enumerate(rows)
    }
    formula_by_row = {int(row): str(formulas[index]) for index, row in enumerate(rows)}
    library_rows = set(map(int, library["rows"]))
    missing_library = library_rows - set(row_to_index)
    if missing_library:
        # Candidate-graph validation needs metadata for the complete sealed library,
        # including rows not reached by any query.
        extra = np.asarray(sorted(missing_library), dtype=np.int64)
        with h5py.File(args.data, "r") as handle:
            extra_ik = text_values(handle["INCHIKEY"], extra)
            extra_adduct = text_values(handle["adduct"], extra)
            extra_mass = numeric_values(handle["precursor_mz"], extra).astype(np.float64)
        row_metadata.update({
            int(row): (str(extra_ik[index])[:14], str(extra_adduct[index]), float(extra_mass[index]))
            for index, row in enumerate(extra)
        })
    validate_candidate_graphs(manifests, library, row_metadata, ppm)
    print(f"[preflight] provenance and {sum(len(x['queries']) for x in manifests.values()):,} candidate graphs PASS")
    if args.preflight_only:
        print("[preflight-only] PASS; no embeddings, features, scores or result files were produced")
        return

    with h5py.File(args.data, "r") as handle:
        spectra = numeric_values(handle["spectrum"], rows)
    embeddings = load_or_build_embeddings(args, rows, spectra, precursor)
    panel_results = {}
    all_records = []
    pair_cache: dict[tuple[int, int], dict[str, float]] = {}
    for panel_index, panel in enumerate(PANEL_FILES):
        summary, records = evaluate_panel(
            panel, manifests[panel], row_to_index, embeddings, spectra, precursor,
            ik14, formula_by_row, configuration, args.peak_tolerance,
            pair_cache, args.bootstrap, args.seed + 100 * panel_index,
        )
        panel_results[panel] = summary
        all_records.extend(records)

    primary = panel_results["P3-main-real-pristine"]
    near = panel_results["P3-near-core-real-pristine"]
    primary_delta = primary["p2b_vs_dreams"]["top1_paired_formula_bootstrap"]
    gates = {
        "primary_recall1_formula_ci_positive": primary_delta["ci_low"] > 0.0,
        "primary_mrr_nonnegative": primary["p2b_vs_dreams"]["mrr_paired_formula_bootstrap"]["mean_delta"] >= 0.0,
        "primary_auc_nonnegative": primary["p2b_vs_dreams"]["auc_paired_formula_bootstrap"]["mean_delta"] >= 0.0,
        "primary_corrected_gt_introduced": primary["p2b_vs_dreams"]["corrected"] > primary["p2b_vs_dreams"]["introduced"],
        "near_core_recall1_nonnegative": near["p2b_vs_dreams"]["top1_paired_formula_bootstrap"]["mean_delta"] >= 0.0,
    }
    gates["pass"] = all(gates.values())
    report = {
        "status": "g8r_p2b_p3_passed" if gates["pass"] else "g8r_p2b_p3_failed",
        "artifact_sha256": sha256_file(args.artifact),
        "selection_sha256": sha256_file(args.selection),
        "p3_lock_summary_sha256": sha256_file(args.p3_dir / "p3_lock_summary.json"),
        "hdf5_sha256": lock["hdf5_sha256"],
        "reference_library_sha256": lock["reference_library_sha256"],
        "frozen_configuration": artifact["configuration"],
        "panels": panel_results,
        "gates": gates,
        "claim_boundary": (
            "Only P3-main-real-pristine is primary. Secondary overlapping panels are "
            "reported separately and are not pooled into the primary confidence interval."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".per_query.csv")
    temporary_json = args.output.with_suffix(".tmp.json")
    temporary_csv = csv_path.with_suffix(".tmp.csv")
    temporary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    temporary_csv.replace(csv_path)
    temporary_json.replace(args.output)
    print(json.dumps({"status": report["status"], "primary": primary, "near": near, "gates": gates},
                     ensure_ascii=False, indent=2))
    print(f"[sealed-result] {args.output}")


if __name__ == "__main__":
    main()
