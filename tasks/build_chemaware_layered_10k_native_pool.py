"""Build an auditable 10k+ native DreaMS corpus around qualified chemistry.

The source pool must already contain the protected Phase-A pool as an exact
prefix followed by independently qualified chemical max-boundary triplets.
This script never manufactures more chemical evidence.  It fills the remaining
training budget only with untouched official DreaMS 10-ppm replay events and
records the three strata separately for future ChemAware-RSI use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_dreams_native_triplets import audit_identity_edges
from build_chemaware_max_boundary_native_triplets import PoolWriter, edges
from build_chemaware_sirius_native_triplets import load_npz, verify_phasea_prefix


ROOT = Path(__file__).resolve().parents[1]
LAYERED_OFFICIAL_REPLAY = 9


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phasea-pool", type=Path, required=True)
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-event-ledger", type=Path, required=True)
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument(
        "--dreams-replay-pool", type=Path,
        default=ROOT / "data/e1/e1_train_triplet_pool_10ppm.npz",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--rule-corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-events", type=int, default=12000)
    parser.add_argument("--minimum-total-events", type=int, default=10000)
    parser.add_argument("--minimum-chemical-events", type=int, default=1000)
    parser.add_argument(
        "--minimum-chemical-queries", type=int, default=500,
        help=(
            "Optional diversity gate. Zero records query coverage as a diagnostic "
            "without turning an arbitrary count into a stopping criterion."
        ),
    )
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def append_pool(pool: Mapping[str, np.ndarray], writer: PoolWriter) -> None:
    required = {
        "anchor_idx", "positive_ptr", "positive_idx", "negative_ptr", "negative_idx",
        "source_query", "negative_candidate", "source_tag", "curriculum_role",
    }
    missing = required.difference(pool)
    if missing:
        raise RuntimeError(f"annotated source pool lacks fields: {sorted(missing)}")
    for event in range(len(pool["anchor_idx"])):
        before = len(writer.anchor)
        writer.append(
            int(pool["anchor_idx"][event]), edges(pool, event, "positive"),
            edges(pool, event, "negative"), int(pool["source_query"][event]),
            int(pool["negative_candidate"][event]), int(pool["source_tag"][event]),
            int(pool["curriculum_role"][event]),
        )
        if len(writer.anchor) == before:
            raise RuntimeError(f"source pool contains duplicate event signature: {event}")


def append_replay(
    replay: Mapping[str, np.ndarray], writer: PoolWriter, target: int, seed: int,
) -> tuple[int, int]:
    required = {"anchor_idx", "positive_ptr", "positive_idx", "negative_ptr", "negative_idx"}
    missing = required.difference(replay)
    if missing:
        raise RuntimeError(f"official replay lacks fields: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    retained = collisions = 0
    for event in rng.permutation(len(replay["anchor_idx"])):
        if len(writer.anchor) >= target:
            break
        before = len(writer.anchor)
        writer.append(
            int(replay["anchor_idx"][event]), edges(replay, int(event), "positive"),
            edges(replay, int(event), "negative"), -1, -1, 0, LAYERED_OFFICIAL_REPLAY,
        )
        retained += int(len(writer.anchor) > before)
        collisions += int(len(writer.anchor) == before)
    return retained, collisions


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.minimum_total_events < 10000:
        raise ValueError("10k corpus gate may not be weakened")
    if args.target_events < args.minimum_total_events:
        raise ValueError("target events are below the formal minimum")
    if args.minimum_chemical_events < 1 or args.minimum_chemical_queries < 0:
        raise ValueError("chemical-event gate must be positive and query gate nonnegative")

    phasea = load_npz(args.phasea_pool)
    source = load_npz(args.source_pool)
    replay = load_npz(args.dreams_replay_pool)
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    source_status = source_report.get("status")
    accepted_source_statuses = {
        "CHEMAWARE_MULTISOURCE_NATIVE_TRIPLETS_COMPLETE",
        "CHEMAWARE_DYNAMIC_REFERENCE_NATIVE_TRIPLETS_COMPLETE",
    }
    if source_status not in accepted_source_statuses:
        raise RuntimeError("source triplet report is not complete")
    if not verify_phasea_prefix(phasea, source):
        raise RuntimeError("qualified source pool does not preserve Phase-A exactly")
    phasea_events = len(phasea["anchor_idx"])
    chemical_events = len(source["anchor_idx"]) - phasea_events
    if source_status == "CHEMAWARE_DYNAMIC_REFERENCE_NATIVE_TRIPLETS_COMPLETE":
        reported_events = int(source_report.get("dynamic_chemical_events_added", -1))
        chemical_queries = int(source_report.get("dynamic_chemical_queries", -1))
        native_tuple = (
            "one q, one qualified active true reference p, and one active false "
            "reference n from a distinct false candidate"
        )
    else:
        reported_events = int(source_report.get("source_events_added", -1))
        chemical_queries = int(source_report.get("source_event_queries", -1))
        native_tuple = "one q, exact current max-reference p*, exact selected max-reference n*"
    if chemical_events != reported_events:
        raise RuntimeError("source report/event cardinality drift")
    if chemical_events < args.minimum_chemical_events:
        raise RuntimeError(
            f"only {chemical_events} qualified chemical events; "
            f"minimum is {args.minimum_chemical_events}"
        )
    if args.minimum_chemical_queries and chemical_queries < args.minimum_chemical_queries:
        raise RuntimeError(
            f"only {chemical_queries} chemical queries; "
            f"minimum is {args.minimum_chemical_queries}"
        )

    writer = PoolWriter()
    append_pool(source, writer)
    source_events = len(writer.anchor)
    replay_retained, replay_collisions = append_replay(
        replay, writer, args.target_events, args.seed,
    )
    output = writer.arrays()
    if len(output["anchor_idx"]) < args.target_events:
        raise RuntimeError(
            f"official replay exhausted at {len(output['anchor_idx'])} events; "
            f"target is {args.target_events}"
        )

    event_rows = []
    for event in range(len(output["anchor_idx"])):
        if event < phasea_events:
            stratum = "protected_phasea"
        elif event < source_events:
            stratum = "qualified_chemical_boundary"
        else:
            stratum = "official_dreams_10ppm_replay"
        event_rows.append({
            "pool_event": event,
            "stratum": stratum,
            "anchor_idx": int(output["anchor_idx"][event]),
            "positive_idx": ";".join(map(str, edges(output, event, "positive"))),
            "negative_idx": ";".join(map(str, edges(output, event, "negative"))),
            "manifest_query": int(output["source_query"][event]),
            "negative_candidate": int(output["negative_candidate"][event]),
            "source_tag": int(output["source_tag"][event]),
            "curriculum_role": int(output["curriculum_role"][event]),
            "loss_weight": "1",
        })

    gates = {
        "source_pool_is_exact_prefix": verify_phasea_prefix(source, output),
        "protected_phasea_is_exact_prefix": verify_phasea_prefix(phasea, output),
        "minimum_total_events": len(output["anchor_idx"]) >= args.minimum_total_events,
        "minimum_qualified_chemical_events": chemical_events >= args.minimum_chemical_events,
        "minimum_qualified_chemical_queries": (
            args.minimum_chemical_queries == 0
            or chemical_queries >= args.minimum_chemical_queries
        ),
        "exact_target_events": len(output["anchor_idx"]) == args.target_events,
        "unmodified_native_triplet_weights": all(row["loss_weight"] == "1" for row in event_rows),
    }
    if not all(gates.values()):
        raise RuntimeError(f"layered 10k corpus gates failed: {gates}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_layered_10k_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **output)
        shutil.copy2(args.validation_pool, temporary / "val_pool.npz")
        shutil.copy2(args.source_event_ledger, temporary / "chemical_source_event_ledger.tsv")
        with (temporary / "corpus_event_manifest.tsv").open(
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer_csv = csv.DictWriter(
                handle, fieldnames=list(event_rows[0]), delimiter="\t", lineterminator="\n",
            )
            writer_csv.writeheader()
            writer_csv.writerows(event_rows)
        rule_corpus = None
        if args.rule_corpus is not None:
            if not args.rule_corpus.is_dir() or not (args.rule_corpus / "report.json").is_file():
                raise RuntimeError("rule corpus does not contain report.json")
            shutil.copytree(args.rule_corpus, temporary / "rule_corpus")
            rule_corpus = {
                "path": str(args.rule_corpus.resolve()),
                "report_sha256": sha256(args.rule_corpus / "report.json"),
            }
        report = {
            "status": "CHEMAWARE_LAYERED_10K_NATIVE_CORPUS_COMPLETE",
            "events": len(output["anchor_idx"]),
            "strata": {
                "protected_phasea": phasea_events,
                "qualified_chemical_boundary": chemical_events,
                "official_dreams_10ppm_replay_added": replay_retained,
            },
            "qualified_chemical_queries": chemical_queries,
            "chemical_query_gate_enabled": args.minimum_chemical_queries > 0,
            "minimum_chemical_queries_requested": args.minimum_chemical_queries,
            "official_replay_collisions_skipped": replay_collisions,
            "rule_corpus": rule_corpus,
            "identity_audit": audit_identity_edges(output, args.data),
            "gates": gates,
            "scientific_contract": {
                "native_tuple": native_tuple,
                "chemical_claim": "only qualified_chemical_boundary events are novel chemical supervision",
                "replay_claim": "official replay raises corpus coverage but is not counted as chemical evidence",
                "loss": "unmodified DreaMS native triplet loss; every event weight is one",
                "sampling": "no copied events, no identity broadcast, no synthetic peak supervision",
                "future_use": "event-level provenance retained for ChemAware-RSI and later ML studies",
                "source_status": source_status,
            },
        }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
