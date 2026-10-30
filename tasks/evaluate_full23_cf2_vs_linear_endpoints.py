#!/usr/bin/env python
"""Read-only paired endpoint evaluation of CF2 against the linear parent-qrel run.

Both inputs are completed sequential-outer-fit development runs.  This entry
point never refits either model.  Before reading either score vector it binds
the two artifacts to the same frozen query/candidate axis by requiring exact
query metadata and exact frozen-evidence fingerprints (Full23 v1, the rebuilt
outer parent, WSE, official DreaMS, Noise, and P2b).  It additionally checks
the axis against the source ledgers used by the generic endpoint evaluator.

The reported endpoints deliberately reuse the established Full23 evaluator:
four panel-level paired reports, source-level query-balanced views, the GNPS
identity IK14 sensitivity view, formula-cluster bootstrap intervals, MRR,
near-query performance, and candidate/query AUC summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from gnps_pair_score_cache import sha256_file
from train_enveda_unified_candidate_integrator import PANELS, WSE, PanelData
from train_full23_centered_residual_from_frozen_ledger import PARENT
from train_full23_parent_query_relative_mgda_residual import (
    METHOD as LINEAR_METHOD,
    SEQUENTIAL_PARENT,
)
from train_full23_parent_qrel_cf2_mgda_residual import METHOD as CF2_METHOD
from evaluate_full23_multisource_additive_endpoints import (
    SOURCE_PREFIX,
    _collapse_records,
    _collapsed_report,
    _metadata,
    _paired_panel_report,
)


SOURCES = ("enveda", "gnps")
FROZEN_AXIS_SCORES = (
    SEQUENTIAL_PARENT,
    PARENT,
    WSE,
    "official_dreams",
    "noise_v1",
    "p2b_noise_v1_frozen",
)
EXTERNAL_SCORE_NAME = {
    PARENT: "additive_linear_control",
    WSE: WSE,
    "official_dreams": "official_dreams",
    "noise_v1": "noise_v1",
    "p2b_noise_v1_frozen": "p2b_noise_v1_frozen",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cf2-run", type=Path, required=True)
    parser.add_argument("--linear-run", type=Path, required=True)
    parser.add_argument("--enveda-ledger", type=Path, required=True)
    parser.add_argument("--gnps-score-dir", type=Path, required=True)
    parser.add_argument("--gnps-axis-ledger", type=Path, required=True)
    parser.add_argument("--gnps-identity-panel", type=Path, required=True)
    parser.add_argument("--gnps-formula-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20261029)
    return parser.parse_args()


def _load_run(run: Path, expected_method: str) -> tuple[dict, Path]:
    report_path = run / "report.json"
    prediction_path = run / "candidate_predictions_outer_oof.npz"
    if not report_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(
            f"completed run lacks report or outer-OOF predictions: {run}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = str(report.get("status", ""))
    if not status.endswith("_COMPLETE"):
        raise RuntimeError(f"run is not complete: {run}: {status!r}")
    if str(report.get("method", "")) != expected_method:
        raise RuntimeError(
            f"run method mismatch: {run}: {report.get('method')!r} != {expected_method!r}"
        )
    recorded = report.get("artifacts", {}).get("predictions_sha256")
    actual = sha256_file(prediction_path)
    if recorded is not None and str(recorded) != actual:
        raise RuntimeError(f"prediction hash disagrees with run report: {run}")
    return report, prediction_path


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _hash_array(digest: "hashlib._Hash", name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    if array.dtype.kind in "USO":
        for item in array.astype("U").ravel():
            encoded = str(item).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        digest.update(np.ascontiguousarray(array).tobytes())


def _external_score(
    source: str,
    panel_name: str,
    name: str,
    enveda: np.lib.npyio.NpzFile,
    gnps_scores: dict[str, np.lib.npyio.NpzFile],
) -> np.ndarray:
    external_name = EXTERNAL_SCORE_NAME[name]
    if source == "enveda":
        key = SOURCE_PREFIX[source][panel_name] + external_name
        body = enveda
    else:
        key = external_name
        body = gnps_scores[panel_name]
    if key not in body.files:
        raise RuntimeError(
            f"external frozen axis lacks {source}/{panel_name}/{external_name}"
        )
    return np.asarray(body[key])


def _aligned_panel(
    cf2: np.lib.npyio.NpzFile,
    linear: np.lib.npyio.NpzFile,
    source: str,
    panel_name: str,
    query_keys: np.ndarray,
    near: np.ndarray,
    formulas: np.ndarray,
    enveda: np.lib.npyio.NpzFile,
    gnps_scores: dict[str, np.lib.npyio.NpzFile],
) -> tuple[PanelData, dict[str, np.ndarray], dict]:
    prefix = f"{source}__{panel_name}__"
    metadata_names = ("query_ptr", "labels", "query_keys", "formula", "near")
    missing = [
        prefix + name for name in metadata_names
        if prefix + name not in cf2.files or prefix + name not in linear.files
    ]
    required_scores = (CF2_METHOD, LINEAR_METHOD, *FROZEN_AXIS_SCORES)
    missing.extend(
        prefix + name for name in required_scores
        if prefix + name not in cf2.files and name != LINEAR_METHOD
        or prefix + name not in linear.files and name != CF2_METHOD
    )
    if missing:
        raise RuntimeError(f"paired artifacts lack required arrays: {sorted(set(missing))}")

    cf2_meta = {name: np.asarray(cf2[prefix + name]) for name in metadata_names}
    linear_meta = {name: np.asarray(linear[prefix + name]) for name in metadata_names}
    checks = {
        f"paired_{name}": _array_equal(cf2_meta[name], linear_meta[name])
        for name in metadata_names
    }
    checks.update({
        "query_keys_external": _array_equal(cf2_meta["query_keys"].astype("U"), query_keys),
        "formula_external": _array_equal(cf2_meta["formula"].astype("U"), formulas),
        "near_external": _array_equal(cf2_meta["near"].astype(bool), near),
    })
    if source == "enveda":
        raw_prefix = SOURCE_PREFIX[source][panel_name]
        raw_ptr = np.asarray(enveda[raw_prefix + "query_ptr"], dtype=np.int64)
        raw_labels = np.asarray(enveda[raw_prefix + "labels"], dtype=np.int8)
    else:
        raw = gnps_scores[panel_name]
        raw_ptr = np.asarray(raw["query_ptr"], dtype=np.int64)
        raw_labels = np.asarray(raw["molecule_label"], dtype=np.int8)
    checks.update({
        "query_ptr_external": _array_equal(cf2_meta["query_ptr"], raw_ptr),
        "labels_external": _array_equal(cf2_meta["labels"], raw_labels),
    })

    shared_scores: dict[str, np.ndarray] = {}
    for name in FROZEN_AXIS_SCORES:
        cf2_score = np.asarray(cf2[prefix + name])
        linear_score = np.asarray(linear[prefix + name])
        checks[f"paired_frozen_score__{name}"] = _array_equal(cf2_score, linear_score)
        shared_scores[name] = cf2_score
        if name in EXTERNAL_SCORE_NAME:
            checks[f"external_frozen_score__{name}"] = _array_equal(
                cf2_score,
                _external_score(source, panel_name, name, enveda, gnps_scores),
            )
    if not all(checks.values()):
        raise RuntimeError(
            f"query/candidate axis mismatch for {source}/{panel_name}: "
            f"{ {key: value for key, value in checks.items() if not value} }"
        )

    ptr = cf2_meta["query_ptr"].astype(np.int64)
    labels = cf2_meta["labels"].astype(np.int8)
    if (
        ptr.ndim != 1 or len(ptr) != len(query_keys) + 1 or ptr[0] != 0
        or ptr[-1] != len(labels) or np.any(np.diff(ptr) < 2)
    ):
        raise RuntimeError(f"invalid complete candidate blocks: {source}/{panel_name}")
    if any(len(value) != len(labels) for value in shared_scores.values()):
        raise RuntimeError(f"frozen candidate score length drifted: {source}/{panel_name}")

    methods = {
        CF2_METHOD: np.asarray(cf2[prefix + CF2_METHOD], dtype=np.float32),
        LINEAR_METHOD: np.asarray(linear[prefix + LINEAR_METHOD], dtype=np.float32),
    }
    if any(len(value) != len(labels) or np.any(~np.isfinite(value)) for value in methods.values()):
        raise RuntimeError(f"paired method scores are incomplete: {source}/{panel_name}")

    # The run artifacts do not carry a separate molecule-ID vector.  The
    # following digest therefore binds each candidate position explicitly to
    # its query block, truth label, and six immutable frozen score vectors,
    # five of which are also checked against the independent source ledger.
    digest = hashlib.sha256()
    for name, value in (
        ("query_ptr", ptr), ("query_keys", query_keys), ("labels", labels),
        ("formula", formulas), ("near", near),
        *((f"frozen::{name}", shared_scores[name]) for name in FROZEN_AXIS_SCORES),
    ):
        _hash_array(digest, name, value)

    panel = PanelData(
        name=f"{source}__{panel_name}", graph=None,
        scalar=np.empty((len(labels), 0), dtype=np.float32), feature_names=(),
        labels=labels, query_ptr=ptr, formulas=formulas.astype("U"),
        query_keys=query_keys.astype("U"), near=near.astype(bool),
        query_weights=np.ones(len(query_keys), dtype=np.float64), scores=methods,
    )
    axis_report = {
        "queries": int(len(query_keys)),
        "candidate_rows": int(len(labels)),
        "candidate_axis_sha256": digest.hexdigest(),
        "exact_checks": checks,
        "semantics": (
            "Exact query metadata plus six shared frozen score vectors bind every "
            "within-query candidate position; five vectors are independently replayed "
            "against the source ledger. No length-only alignment is accepted."
        ),
    }
    return panel, methods, axis_report


def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap-resamples must be at least 100")
    cf2_report, cf2_path = _load_run(args.cf2_run, CF2_METHOD)
    linear_report, linear_path = _load_run(args.linear_run, LINEAR_METHOD)

    gnps_paths = {
        name: args.gnps_score_dir / f"candidate_scores_{name}.npz" for name in PANELS
    }
    gnps_panel_paths = {
        "identity_disjoint": args.gnps_identity_panel,
        "formula_disjoint": args.gnps_formula_panel,
    }
    required = (
        args.enveda_ledger, args.gnps_axis_ledger,
        *gnps_paths.values(), *gnps_panel_paths.values(),
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    panel_reports: dict[str, dict] = {source: {} for source in SOURCES}
    source_panels: dict[str, dict[str, PanelData]] = {source: {} for source in SOURCES}
    source_methods: dict[str, dict[str, dict[str, np.ndarray]]] = {
        source: {} for source in SOURCES
    }
    source_keys: dict[str, dict[str, np.ndarray]] = {source: {} for source in SOURCES}
    axis_reports: dict[str, dict] = {source: {} for source in SOURCES}
    gnps_ik14: dict[str, np.ndarray] = {}

    with (
        np.load(cf2_path, allow_pickle=False) as cf2,
        np.load(linear_path, allow_pickle=False) as linear,
        np.load(args.enveda_ledger, allow_pickle=False) as enveda,
        np.load(args.gnps_axis_ledger, allow_pickle=False) as axis,
        np.load(gnps_paths["identity_disjoint"], allow_pickle=False) as gnps_identity,
        np.load(gnps_paths["formula_disjoint"], allow_pickle=False) as gnps_formula,
        np.load(gnps_panel_paths["identity_disjoint"], allow_pickle=False) as identity_panel,
        np.load(gnps_panel_paths["formula_disjoint"], allow_pickle=False) as formula_panel,
    ):
        gnps_scores = {
            "identity_disjoint": gnps_identity,
            "formula_disjoint": gnps_formula,
        }
        gnps_panels = {
            "identity_disjoint": identity_panel,
            "formula_disjoint": formula_panel,
        }
        for source_index, source in enumerate(SOURCES):
            for panel_index, panel_name in enumerate(PANELS):
                keys, near, formulas, ik14 = _metadata(
                    source, panel_name, enveda, gnps_scores, axis, gnps_panels,
                )
                panel, methods, axis_report = _aligned_panel(
                    cf2, linear, source, panel_name, keys, near, formulas,
                    enveda, gnps_scores,
                )
                source_panels[source][panel_name] = panel
                source_methods[source][panel_name] = methods
                source_keys[source][panel_name] = keys
                axis_reports[source][panel_name] = axis_report
                if ik14 is not None:
                    gnps_ik14[panel_name] = ik14
                panel_reports[source][panel_name] = _paired_panel_report(
                    panel, methods, CF2_METHOD, LINEAR_METHOD,
                    args.bootstrap_resamples,
                    args.seed + source_index * 10_000 + panel_index * 100,
                )

    query_balanced = {}
    for source_index, source in enumerate(SOURCES):
        collapsed, formulas = _collapse_records(
            source_panels[source], source_methods[source], source_keys[source],
            CF2_METHOD, LINEAR_METHOD,
        )
        overlap = len(
            set(map(str, source_keys[source]["identity_disjoint"]))
            & set(map(str, source_keys[source]["formula_disjoint"]))
        )
        query_balanced[source] = {
            "panel_occurrences": int(sum(len(v) for v in source_keys[source].values())),
            "cross_panel_repeated_queries": int(overlap),
            **_collapsed_report(
                collapsed, formulas, CF2_METHOD, LINEAR_METHOD,
                args.bootstrap_resamples, args.seed + 30_000 + source_index * 100,
                (
                    "Each stable source query key has total mass one; a key present in "
                    "both panels contributes the mean of its panel-specific outcomes."
                ),
            ),
        }

    ik14_collapsed, ik14_formulas = _collapse_records(
        {"identity_disjoint": source_panels["gnps"]["identity_disjoint"]},
        {"identity_disjoint": source_methods["gnps"]["identity_disjoint"]},
        {"identity_disjoint": gnps_ik14["identity_disjoint"]},
        CF2_METHOD, LINEAR_METHOD,
    )
    ik14_report = {
        "spectrum_queries": int(len(gnps_ik14["identity_disjoint"])),
        **_collapsed_report(
            ik14_collapsed, ik14_formulas, CF2_METHOD, LINEAR_METHOD,
            args.bootstrap_resamples, args.seed + 40_000,
            (
                "GNPS identity spectra are grouped by aligned 14-character query "
                "InChIKey block; each IK14 has total mass one."
            ),
        ),
    }

    report = {
        "status": "FULL23_CF2_VS_LINEAR_ENDPOINT_EVALUATION_COMPLETE",
        "candidate": CF2_METHOD,
        "baseline": LINEAR_METHOD,
        "dataset_role": "consumed multisource development outer-OOF only",
        "allow_final_external_claim": False,
        "candidate_axis_contract": axis_reports,
        "panels": panel_reports,
        "query_balanced_primary_endpoints": query_balanced,
        "gnps_identity_ik14_collapse_sensitivity": ik14_report,
        "inputs": {
            "cf2_status": cf2_report["status"],
            "cf2_report_sha256": sha256_file(args.cf2_run / "report.json"),
            "cf2_predictions_sha256": sha256_file(cf2_path),
            "linear_status": linear_report["status"],
            "linear_report_sha256": sha256_file(args.linear_run / "report.json"),
            "linear_predictions_sha256": sha256_file(linear_path),
            "enveda_ledger_sha256": sha256_file(args.enveda_ledger),
            "gnps_axis_ledger_sha256": sha256_file(args.gnps_axis_ledger),
            **{
                f"gnps_{name}_scores_sha256": sha256_file(path)
                for name, path in gnps_paths.items()
            },
            **{
                f"gnps_{name}_panel_sha256": sha256_file(path)
                for name, path in gnps_panel_paths.items()
            },
        },
        "claim_boundary": (
            "CF2 and linear are both sequential outer-fit estimates on consumed Enveda "
            "and GNPS development resources. This paired read-only comparison cannot "
            "support an untouched external claim."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main() -> None:
    args = arguments()
    report = evaluate(args)
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
