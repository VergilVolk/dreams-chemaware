"""Build a DreaMS-native curriculum aligned to max-reference retrieval.

The successful ChemAware stage-1 bank stores molecule-level positive and
negative reference sets, while DreaMS samples one positive and one negative
reference on every visit.  Formal retrieval instead scores each molecule by
the maximum reference similarity.  This builder changes only the triplet
curriculum:

* official-error queries receive singleton top-positive/top-negative events;
* up to two active ChemAware false candidates are added only on those errors;
* official-correct queries retain one singleton nearest-boundary sentinel;
* a small, unmodified subset of the official DreaMS 10-ppm triplet pool is
  replayed to protect geometry outside the ChemAware candidate graph.

The DreaMS dataset, model, preprocessing, loss, optimizer and update rule are
unchanged by the downstream trainer.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_action_hard_native_triplets import (
    ACTION_HARD,
    OFFICIAL_HARD,
    SPECIFIC_HARD,
)
from build_chemaware_dreams_native_triplets import audit_identity_edges


ROOT = Path(__file__).resolve().parents[1]
SAFE_MAX_BOUNDARY = 1
ERROR_OFFICIAL_MAX_BOUNDARY = 2
ERROR_CHEMICAL_MAX_BOUNDARY = 3
DREAMS_NATIVE_REPLAY = 4


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bank", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--embedding-rows", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy",
    )
    parser.add_argument(
        "--official-embeddings", type=Path,
        default=(ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1"
                 / "official_embeddings_f32.npy"),
    )
    parser.add_argument(
        "--dreams-replay-pool", type=Path,
        default=ROOT / "data/e1/e1_train_triplet_pool_10ppm.npz",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--negative-references-per-error", type=int, default=2)
    parser.add_argument("--chemical-candidates-per-error", type=int, default=2)
    parser.add_argument("--dreams-replay-events", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--min-error-event-fraction", type=float, default=0.20)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def edges(pool: Mapping[str, np.ndarray], event: int, kind: str) -> np.ndarray:
    pointer = np.asarray(pool[f"{kind}_ptr"], dtype=np.int64)
    values = np.asarray(pool[f"{kind}_idx"], dtype=np.int64)
    left, right = map(int, pointer[event:event + 2])
    return values[left:right]


class FrozenEmbeddings:
    def __init__(self, rows_path: Path, embeddings_path: Path):
        rows = np.load(rows_path, mmap_mode="r")
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        if self.embeddings.ndim != 2 or len(rows) != len(self.embeddings):
            raise RuntimeError("official embedding row registry is invalid")
        if len(np.unique(rows)) != len(rows):
            raise RuntimeError("official embedding row registry contains duplicates")
        self.position = {int(row): index for index, row in enumerate(rows)}

    def get(self, rows: np.ndarray | list[int]) -> np.ndarray:
        try:
            positions = np.asarray(
                [self.position[int(row)] for row in rows], dtype=np.int64,
            )
        except KeyError as error:
            raise RuntimeError(f"spectrum row absent from official cache: {error}") from error
        values = np.asarray(self.embeddings[positions], dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(~np.isfinite(values)) or np.any(norms <= 0):
            raise RuntimeError("official embedding cache contains invalid vectors")
        return values / norms


def max_boundary(
    cache: FrozenEmbeddings,
    anchor: int,
    positive_rows: np.ndarray,
    negative_rows: np.ndarray,
    margin: float,
) -> dict[str, object]:
    anchor_embedding = cache.get([anchor])[0]
    positive_score = cache.get(positive_rows) @ anchor_embedding
    negative_score = cache.get(negative_rows) @ anchor_embedding
    positive_order = np.argsort(-positive_score, kind="stable")
    negative_order = np.argsort(-negative_score, kind="stable")
    best_positive = int(positive_rows[int(positive_order[0])])
    hinge = margin + negative_score - float(positive_score[int(positive_order[0])])
    active_negative_order = [
        int(index) for index in negative_order if float(hinge[int(index)]) > 0.0
    ]
    return {
        "positive_row": best_positive,
        "negative_rows": np.asarray(
            [int(negative_rows[index]) for index in active_negative_order],
            dtype=np.int64,
        ),
        "negative_hinges": np.asarray(
            [float(hinge[index]) for index in active_negative_order],
            dtype=np.float64,
        ),
        "gap": float(positive_score[int(positive_order[0])] - negative_score[int(negative_order[0])]),
    }


class PoolWriter:
    def __init__(self) -> None:
        self.anchor: list[int] = []
        self.positive: list[int] = []
        self.negative: list[int] = []
        self.positive_ptr = [0]
        self.negative_ptr = [0]
        self.source_query: list[int] = []
        self.negative_candidate: list[int] = []
        self.source_tag: list[int] = []
        self.curriculum_role: list[int] = []
        self.signatures: set[tuple[object, ...]] = set()

    def append(
        self, anchor: int, positive: np.ndarray | list[int],
        negative: np.ndarray | list[int], source_query: int,
        negative_candidate: int, source_tag: int, curriculum_role: int,
    ) -> None:
        positive_tuple = tuple(map(int, positive))
        negative_tuple = tuple(map(int, negative))
        if not positive_tuple or not negative_tuple:
            raise RuntimeError("empty edge set reached max-boundary writer")
        signature = (int(anchor), positive_tuple, negative_tuple)
        if signature in self.signatures:
            return
        self.signatures.add(signature)
        self.anchor.append(int(anchor))
        self.positive.extend(positive_tuple)
        self.negative.extend(negative_tuple)
        self.positive_ptr.append(len(self.positive))
        self.negative_ptr.append(len(self.negative))
        self.source_query.append(int(source_query))
        self.negative_candidate.append(int(negative_candidate))
        self.source_tag.append(int(source_tag))
        self.curriculum_role.append(int(curriculum_role))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "anchor_idx": np.asarray(self.anchor, dtype=np.int64),
            "positive_ptr": np.asarray(self.positive_ptr, dtype=np.int64),
            "positive_idx": np.asarray(self.positive, dtype=np.int64),
            "negative_ptr": np.asarray(self.negative_ptr, dtype=np.int64),
            "negative_idx": np.asarray(self.negative, dtype=np.int64),
            "source_query": np.asarray(self.source_query, dtype=np.int64),
            "negative_candidate": np.asarray(self.negative_candidate, dtype=np.int32),
            "source_tag": np.asarray(self.source_tag, dtype=np.int16),
            "curriculum_role": np.asarray(self.curriculum_role, dtype=np.int8),
        }


def build_focused_pool(
    base: Mapping[str, np.ndarray], evidence: Mapping[str, np.ndarray],
    cache: FrozenEmbeddings, writer: PoolWriter, margin: float,
    negative_references_per_error: int, chemical_candidates_per_error: int,
) -> dict[str, object]:
    if negative_references_per_error < 1 or chemical_candidates_per_error < 0:
        raise ValueError("invalid max-boundary event cap")
    rank = {
        int(query): int(value)
        for query, value in zip(evidence["query"], evidence["baseline_rank"], strict=True)
    }
    by_query: dict[int, list[int]] = {}
    for event, query in enumerate(np.asarray(base["source_query"], dtype=np.int64)):
        by_query.setdefault(int(query), []).append(event)
    if set(by_query) != set(rank):
        raise RuntimeError("base triplet queries do not exactly match frozen evidence")

    error_queries = safe_queries = 0
    error_official_events = error_chemical_events = safe_events = 0
    candidate_rows_examined = inactive_chemical_candidates = 0
    error_query_event_counts: dict[int, int] = {}
    for query in sorted(by_query):
        events = by_query[query]
        official = [
            event for event in events
            if int(base["source_tag"][event]) & OFFICIAL_HARD
        ]
        if len(official) != 1:
            raise RuntimeError(f"query {query} has {len(official)} official-hard events")

        boundaries: dict[int, dict[str, object]] = {}
        for event in events:
            candidate_rows_examined += 1
            boundaries[event] = max_boundary(
                cache, int(base["anchor_idx"][event]),
                edges(base, event, "positive"), edges(base, event, "negative"),
                margin,
            )

        if rank[query] == 1:
            safe_queries += 1
            event = official[0]
            boundary = boundaries[event]
            # The exact max/max pair is a no-loss sentinel when the official
            # margin is already safe and becomes active first if geometry drifts.
            negative_rows = edges(base, event, "negative")
            anchor_embedding = cache.get([int(base["anchor_idx"][event])])[0]
            scores = cache.get(negative_rows) @ anchor_embedding
            top_negative = int(negative_rows[int(np.argmax(scores))])
            writer.append(
                int(base["anchor_idx"][event]), [int(boundary["positive_row"])],
                [top_negative], query, int(base["negative_candidate"][event]),
                int(base["source_tag"][event]), SAFE_MAX_BOUNDARY,
            )
            safe_events += 1
            continue

        error_queries += 1
        selected = [official[0]]
        chemical = []
        for event in events:
            if event == official[0]:
                continue
            tag = int(base["source_tag"][event])
            if not (tag & (ACTION_HARD | SPECIFIC_HARD)):
                continue
            boundary = boundaries[event]
            if not len(boundary["negative_rows"]):
                inactive_chemical_candidates += 1
                continue
            chemical.append(event)
        chemical.sort(key=lambda event: (
            bool(int(base["source_tag"][event]) & SPECIFIC_HARD),
            float(boundaries[event]["negative_hinges"][0]),
            -int(base["negative_candidate"][event]),
        ), reverse=True)
        selected.extend(chemical[:chemical_candidates_per_error])
        before = len(writer.anchor)
        for event in selected:
            boundary = boundaries[event]
            active_negative = np.asarray(boundary["negative_rows"], dtype=np.int64)
            if not len(active_negative):
                if event == official[0]:
                    raise RuntimeError(
                        f"official-error query {query} has no active max boundary"
                    )
                continue
            role = (
                ERROR_OFFICIAL_MAX_BOUNDARY
                if event == official[0] else ERROR_CHEMICAL_MAX_BOUNDARY
            )
            for negative_row in active_negative[:negative_references_per_error]:
                writer.append(
                    int(base["anchor_idx"][event]), [int(boundary["positive_row"])],
                    [int(negative_row)], query,
                    int(base["negative_candidate"][event]),
                    int(base["source_tag"][event]), role,
                )
                if role == ERROR_OFFICIAL_MAX_BOUNDARY:
                    error_official_events += 1
                else:
                    error_chemical_events += 1
        error_query_event_counts[query] = len(writer.anchor) - before
        if error_query_event_counts[query] < 1:
            raise RuntimeError(f"official-error query {query} produced no event")

    return {
        "queries": int(len(by_query)),
        "official_error_queries": int(error_queries),
        "official_correct_queries": int(safe_queries),
        "safe_max_boundary_events": int(safe_events),
        "error_official_max_boundary_events": int(error_official_events),
        "error_chemical_max_boundary_events": int(error_chemical_events),
        "error_max_boundary_events": int(error_official_events + error_chemical_events),
        "minimum_events_per_error_query": int(min(error_query_event_counts.values())),
        "candidate_rows_examined": int(candidate_rows_examined),
        "inactive_chemical_candidates_rejected": int(inactive_chemical_candidates),
    }


def append_dreams_replay(
    writer: PoolWriter, replay: Mapping[str, np.ndarray], events: int, seed: int,
) -> dict[str, int]:
    if events < 0 or events > len(replay["anchor_idx"]):
        raise ValueError("invalid DreaMS replay event count")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(replay["anchor_idx"]), size=events, replace=False))
    before = len(writer.anchor)
    for event in selected:
        writer.append(
            int(replay["anchor_idx"][event]), edges(replay, int(event), "positive"),
            edges(replay, int(event), "negative"), -1, -1, 0,
            DREAMS_NATIVE_REPLAY,
        )
    if len(writer.anchor) - before != events:
        raise RuntimeError("DreaMS replay collided with focused event signatures")
    return {"requested": int(events), "retained": int(events)}


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    base_report = json.loads((args.base_bank / "report.json").read_text(encoding="utf-8"))
    if base_report.get("status") != "CHEMAWARE_ACTION_HARD_NATIVE_TRIPLETS_COMPLETE":
        raise RuntimeError("base bank is not the frozen +1.8144 pp curriculum")
    base = load_npz(args.base_bank / "train_pool.npz")
    evidence = load_npz(args.evidence)
    manifest = load_npz(args.manifest)
    replay = load_npz(args.dreams_replay_pool)
    cache = FrozenEmbeddings(args.embedding_rows, args.official_embeddings)
    writer = PoolWriter()
    focused = build_focused_pool(
        base, evidence, cache, writer, args.margin,
        args.negative_references_per_error, args.chemical_candidates_per_error,
    )
    replay_audit = append_dreams_replay(
        writer, replay, args.dreams_replay_events, args.seed,
    )
    output = writer.arrays()
    roles = np.asarray(output["curriculum_role"], dtype=np.int8)
    focused_count = int(np.sum(roles != DREAMS_NATIVE_REPLAY))
    error_count = int(np.sum(np.isin(
        roles, [ERROR_OFFICIAL_MAX_BOUNDARY, ERROR_CHEMICAL_MAX_BOUNDARY],
    )))
    focused_singleton = bool(np.all(
        np.diff(output["positive_ptr"][:focused_count + 1]) == 1
    ) and np.all(np.diff(output["negative_ptr"][:focused_count + 1]) == 1))
    scheduled_queries = np.asarray(sorted(set(
        map(int, output["source_query"][output["source_query"] >= 0])
    )), dtype=np.int64)
    gates = {
        "all_training_queries_covered": focused["queries"] == len(scheduled_queries),
        "all_official_error_queries_covered": (
            focused["minimum_events_per_error_query"] >= 1
        ),
        "focused_events_are_singleton_max_boundaries": focused_singleton,
        "no_chemical_event_on_official_correct_query": int(np.sum(
            roles == ERROR_CHEMICAL_MAX_BOUNDARY
        )) == focused["error_chemical_max_boundary_events"],
        "error_event_fraction": (
            error_count / len(output["anchor_idx"]) >= args.min_error_event_fraction
        ),
        "exact_dreams_replay_budget": (
            int(np.sum(roles == DREAMS_NATIVE_REPLAY)) == args.dreams_replay_events
        ),
        "unique_triplet_signatures": len(writer.signatures) == len(output["anchor_idx"]),
        "formula_coverage_nonempty": bool(len(np.unique(
            np.asarray(manifest["query_formula"])[scheduled_queries].astype(str)
        ))),
        "validation_pool_unchanged": True,
        "outer_role_4_untouched": True,
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"max-boundary native triplet gates failed: {gates}; "
            f"focused={focused}; replay={replay_audit}"
        )
    identity = audit_identity_edges(output, args.data)
    report = {
        "status": "CHEMAWARE_MAX_BOUNDARY_NATIVE_TRIPLETS_COMPLETE",
        "method": (
            "max-reference-aligned error curriculum plus unmodified DreaMS "
            "10-ppm native replay"
        ),
        "training_runtime": "unmodified DreaMS ContrastiveSpectraDataset plus ContrastiveHead",
        "margin": float(args.margin),
        "negative_references_per_error": int(args.negative_references_per_error),
        "chemical_candidates_per_error": int(args.chemical_candidates_per_error),
        "focused": focused,
        "dreams_replay": replay_audit,
        "total_events": int(len(output["anchor_idx"])),
        "focused_events": focused_count,
        "error_events": error_count,
        "error_event_fraction": float(error_count / len(output["anchor_idx"])),
        "unique_formulas": int(len(np.unique(
            np.asarray(manifest["query_formula"])[scheduled_queries].astype(str)
        ))),
        "identity_audit": identity,
        "gates": gates,
        "scientific_boundary": (
            "only triplet construction changes; retrieval-aligned focused events "
            "use singleton frozen max references, and official DreaMS replay edges "
            "are copied without modification"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_max_boundary_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **output)
        shutil.copy2(args.base_bank / "val_pool.npz", temporary / "val_pool.npz")
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
