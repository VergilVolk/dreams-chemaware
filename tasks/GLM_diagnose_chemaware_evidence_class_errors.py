"""Decompose ChemAware current-error boundaries by source-evidence class.

Scientific purpose
------------------
The fragment-graph/MassBank union continuation fail-closed twice (seed and
frozen Phase-A geometry) with ~511 events, ~6 correctable winners, while the
raw ledger relation pool holds 12,755 (true, false) pairs over 4,032 train
queries.  This diagnostic answers the one question that decides the route's
fate: **where do the current model errors actually sit in the evidence
structure?**

Every (true, false) pair of every ledger query is classified with the exact
qualification semantics of ``build_chemaware_multisource_native_triplets.py``
(``pair_evidence``): a family *supports* only when its directional delta beats
every matched-control delta; a family *opposes* on the raw sign of its delta;
an opposition blocks only when its confidence tier rank is at least the
strongest supporting rank.  The diagnostic adds one clearly-labelled symmetric
extension: an opposition is *dominant* only when the reversed delta also beats
the reversed control deltas -- i.e. the same statistical standard the builder
already imposes on support.  Raw-sign opposition without that significance is
*sub-significant*.

Pair classes (stance-based):

* ``unanimous_admitted_style``       - >=1 dominant support, no blocking oppose
* ``vetoed_by_subsignificant_oppose``- >=1 support, blocking oppose that is not
                                        dominant (recoverable by symmetric
                                        significance)
* ``contested_dominant``             - >=1 support and >=1 dominant blocking
                                        oppose (chemically real conflict)
* ``dominant_opposition_only``       - no support, >=1 dominant oppose
* ``no_dominant_signal``             - families present, nothing dominant

Boundary classes add two populations that must never be conflated with weak
evidence:

* ``no_ledger_coverage``             - the winning false candidate pair has no
                                        jointly-scored family rows at all
* ``ambiguous_tie``                  - multiple false candidates tie at the top
                                        and their pair classes disagree

For each query the frozen-geometry ranking replicates ``candidate_geometry``
(molecule-max over unique candidate reference rows, anchor row excluded from
the true candidate, exactly-one-true enforcement) with the evaluator's
tie-unfavorable policy, so an *error boundary* is the (true, winning-false)
pair of a query whose best false score >= best true score.

Statistics: the null is the fixed-margin hypergeometric expectation
E = errors x (boundary-class query fraction), reported with standardized
residuals.  Because boundary class and error status both cluster by formula,
exchangeability is violated at the query level; the primary significance
statement is therefore a within-formula-cluster label permutation test
(labels permuted only among queries sharing a formula), with the standardized
residuals as descriptive auxiliaries.

The ``GLM_`` file prefix is a mandated repository namespace for this session,
not a claim of fitting generalized linear models; this tool is a contingency
table / permutation diagnostic.

Read-only, CPU-only, truth-blind in selection (identity labels only identify
the true candidate, never as a training signal), touches only ledger-scope
queries, never accesses formula role 4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

# Confidence-tier arbitration, replicated verbatim from
# build_chemaware_multisource_native_triplets.py (line 33): an opposition
# blocks only when its rank >= the strongest supporting rank.
CONFIDENCE_RANKS = {"A": 3, "B": 2, "C": 1, "Q": 0}

STATUS = "GLM_CHEMAWARE_EVIDENCE_CLASS_DIAGNOSTIC_COMPLETE"

PAIR_CLASS_ORDER = (
    "unanimous_admitted_style",
    "vetoed_by_subsignificant_oppose",
    "contested_dominant",
    "dominant_opposition_only",
    "no_dominant_signal",
)
BOUNDARY_ONLY_CLASSES = ("no_ledger_coverage", "ambiguous_tie")
BOUNDARY_CLASS_ORDER = PAIR_CLASS_ORDER + BOUNDARY_ONLY_CLASSES

CLASS_DEFINITIONS = {
    "unanimous_admitted_style": (
        "At least one family is dominantly supporting (directional delta beats "
        "every matched-control delta) and no family with tier rank >= the "
        "strongest support rank opposes on the raw sign. Other families may be "
        "absent, inapplicable, or abstaining: this is admission under builder "
        "rules, NOT a claim that all families significantly support."
    ),
    "vetoed_by_subsignificant_oppose": (
        "At least one dominant support exists, and at least one blocking-tier "
        "family opposes on the raw sign, but every blocking opposition fails "
        "the symmetric dominance standard (reversed delta does not beat the "
        "reversed control deltas). This is the mass a symmetric-significance "
        "rule would recover."
    ),
    "contested_dominant": (
        "At least one dominant support and at least one dominant blocking "
        "opposition: strong evidence points both ways. Not a curriculum by "
        "itself; if used, it must be a separate experimental arm."
    ),
    "dominant_opposition_only": (
        "No dominant support; at least one family dominantly points toward the "
        "false candidate. Structurally hard boundary."
    ),
    "no_dominant_signal": (
        "Families jointly score the pair but neither side dominates any of "
        "them. Evidence exists but carries no reliable direction."
    ),
    "no_ledger_coverage": (
        "Boundary-only class: the winning false candidate pair has no jointly "
        "scored family rows. This is absent evidence, categorically different "
        "from weak evidence, and must be reported separately in any closure "
        "argument."
    ),
    "ambiguous_tie": (
        "Boundary-only class: multiple false candidates tie at the top of the "
        "geometry and their pair classes disagree, so the error boundary "
        "cannot be attributed to one evidence class."
    ),
}

CAVEATS = (
    "massbank_recurrent_product_ion and massbank_recurrent_neutral_loss derive "
    "from the same MassBank ledger and are NOT independent confirmations; any "
    "multi-source agreement statement must not count them twice.",
    "Query-level exchangeability is violated by shared formulas; the "
    "fixed_margin_z values are descriptive and the cluster_permutation_test "
    "block is the primary significance statement.",
    "Counts are train-panel opportunities, not selection-panel corrections; "
    "no fixed conversion factor is assumed or implied.",
    "Any panel-size arithmetic (e.g. winner-to-percentage-point conversion) "
    "must cite the executed role-2 evaluation report's queries field rather "
    "than a hardcoded panel size.",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    # Byte-for-byte replica of the encode/builder array digest so cache
    # reports verify without importing those modules.
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        manifest = {key: np.asarray(loaded[key]) for key in loaded.files}
    for key in (
        "query_ptr", "molecule_ptr", "pair_candidate_row", "molecule_label",
        "query_row", "molecule_formula", "molecule_ik14", "query_formula",
    ):
        if key not in manifest:
            raise RuntimeError(f"candidate manifest lacks required field: {key}")
    return manifest


def read_family_registry(ledger_dir: Path) -> dict[str, dict[str, object]]:
    report_path = ledger_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    table_path = ledger_dir / "candidate_scores.tsv"
    if report.get("status") != "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_COMPLETE":
        raise RuntimeError(f"candidate source ledger is incomplete: {ledger_dir}")
    if report.get("truth_fields_exported") is not False:
        raise RuntimeError(f"candidate source ledger is not truth-blind: {ledger_dir}")
    if report.get("candidate_scores_sha256") != file_sha256(table_path):
        raise RuntimeError(f"candidate source ledger hash drift: {ledger_dir}")
    families = report.get("source_families")
    if not isinstance(families, dict) or not families:
        raise RuntimeError(f"ledger report lacks source_families: {ledger_dir}")
    registry: dict[str, dict[str, object]] = {}
    skipped: list[str] = []
    for name, body in families.items():
        if not bool(body.get("specificity_gate_passed", False)):
            skipped.append(name)
            continue
        tier = str(body.get("confidence_tier", ""))[0]
        if tier not in CONFIDENCE_RANKS:
            raise RuntimeError(f"unknown confidence tier for family {name}: {tier!r}")
        controls = body.get("matched_controls") or []
        scope = str(body.get("scope", ""))
        if scope not in {"all_candidates", "cross_formula", "within_formula"}:
            raise RuntimeError(f"invalid source scope for family {name}: {scope!r}")
        if body.get("larger_is_better") is not True:
            raise RuntimeError(f"unsupported score direction for family {name}")
        registry[name] = {
            "scope": scope,
            "confidence_tier": str(body["confidence_tier"]),
            "tier_rank": CONFIDENCE_RANKS[tier],
            "matched_controls": [str(c) for c in controls],
            "larger_is_better": True,
        }
    if skipped:
        print(f"  note: gate-failed families excluded from {ledger_dir.name}: {skipped}")
    if not registry:
        raise RuntimeError(f"ledger has no gate-passed family: {ledger_dir}")
    return registry


def optional_float(raw: str) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    return float(text)


def read_ledger_scores(
    ledger_dir: Path, registry: dict[str, dict[str, object]],
) -> dict[int, dict[int, dict[str, dict[str, object]]]]:
    """Parse candidate_scores.tsv exactly like read_source_ledgers.

    Control columns are discovered from the header by the ``control_*_score``
    pattern so heterogeneous ledgers (MassBank, fragment-graph) parse
    identically without misreading unrelated control_* fields.
    """
    tsv_path = ledger_dir / "candidate_scores.tsv"
    if not tsv_path.is_file():
        raise FileNotFoundError(tsv_path)
    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"empty ledger tsv: {tsv_path}")
    header = lines[0].split("\t")
    required = ("manifest_query", "local_candidate", "source_family", "source_score")
    for column in required:
        if column not in header:
            raise RuntimeError(f"ledger tsv lacks column {column}: {tsv_path}")
    control_columns = tuple(
        name for name in header
        if name.startswith("control_") and name.endswith("_score")
    )
    scores: dict[int, dict[int, dict[str, dict[str, object]]]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        row = dict(zip(header, line.split("\t"), strict=True))
        family = row["source_family"]
        if family not in registry:
            continue
        query = int(row["manifest_query"])
        candidate = int(row["local_candidate"])
        score = optional_float(row.get("source_score", ""))
        control_values = tuple(
            optional_float(row.get(name, "")) for name in control_columns
        )
        body = {
            "ik14": str(row.get("ik14", "")),
            "formula": str(row.get("formula", "")),
            "scope": str(row.get("scope", "all_candidates")),
            "score": score,
            "control_scores": control_values,
            "control_names": control_columns,
            "controls": bool(int(row.get("controls_available", "0") or 0)),
        }
        slot = scores.setdefault(query, {}).setdefault(candidate, {})
        if family in slot:
            raise RuntimeError(
                f"duplicate source score: query={query} candidate={candidate} "
                f"family={family}"
            )
        slot[family] = body
    if not scores:
        raise RuntimeError(f"ledger produced no usable scores: {tsv_path}")
    return scores


def merge_ledger_scores(
    target: dict[int, dict[int, dict[str, dict[str, object]]]],
    incoming: Mapping[int, Mapping[int, Mapping[str, dict[str, object]]]],
) -> None:
    """Union independent source families on the same query/candidate keys."""
    for query, candidates in incoming.items():
        query_slot = target.setdefault(int(query), {})
        for candidate, families in candidates.items():
            candidate_slot = query_slot.setdefault(int(candidate), {})
            for family, body in families.items():
                if family in candidate_slot:
                    raise RuntimeError(
                        "duplicate source family after ledger merge: "
                        f"query={query} candidate={candidate} family={family}"
                    )
                candidate_slot[str(family)] = dict(body)


def validate_scores_against_manifest(
    scores: Mapping[int, Mapping[int, Mapping[str, dict[str, object]]]],
    registry: Mapping[str, Mapping[str, object]],
    manifest: Mapping[str, np.ndarray],
) -> None:
    for query, candidates in scores.items():
        if query < 0 or query >= len(manifest["query_row"]):
            raise RuntimeError(f"source query outside manifest: {query}")
        left, right = map(int, manifest["query_ptr"][query:query + 2])
        for candidate, families in candidates.items():
            if candidate < 0 or candidate >= right - left:
                raise RuntimeError(
                    f"source candidate outside manifest: query={query} candidate={candidate}"
                )
            molecule = left + candidate
            expected_formula = str(manifest["molecule_formula"][molecule])
            expected_ik14 = str(manifest["molecule_ik14"][molecule])
            for family, body in families.items():
                if str(body["scope"]) != str(registry[family]["scope"]):
                    raise RuntimeError(
                        f"source scope drift: query={query} candidate={candidate} family={family}"
                    )
                if str(body["formula"]) != expected_formula:
                    raise RuntimeError(
                        f"source formula drift: query={query} candidate={candidate} family={family}"
                    )
                if str(body["ik14"]) != expected_ik14:
                    raise RuntimeError(
                        f"source IK14 drift: query={query} candidate={candidate} family={family}"
                    )


def family_stance(
    true_body: Mapping[str, object], false_body: Mapping[str, object],
    family: str, registry: dict[str, dict[str, object]],
) -> str:
    """Replicate pair_evidence stance semantics plus the symmetric extension.

    Builder-faithful part (build_chemaware_multisource_native_triplets.py
    pair_evidence): applicable scope; finite scores; ``delta <= 0`` is an
    opposition on the raw sign; a positive delta supports only when controls
    are available on both sides (required for tier C / matched-controls
    families) and the delta beats every control delta.

    Symmetric extension (diagnostic only, clearly not part of the builder):
    an opposition is ``oppose_dominant`` when the reversed delta beats every
    reversed control delta; otherwise it is ``oppose_subsignificant``.
    """
    meta = registry[family]
    true_score, false_score = true_body["score"], false_body["score"]
    if true_score is None or false_score is None:
        return "absent"
    if not (math.isfinite(true_score) and math.isfinite(false_score)):
        return "absent"
    same_formula = str(true_body["formula"]) == str(false_body["formula"])
    scope = str(true_body["scope"])
    applicable = (
        scope == "all_candidates"
        or (scope == "cross_formula" and not same_formula)
        or (scope == "within_formula" and same_formula)
    )
    if not applicable:
        return "inapplicable"
    delta = float(true_score) - float(false_score)
    tier = str(meta["confidence_tier"])[0]
    requires_controls = bool(meta["matched_controls"]) or tier == "C"
    both_controls = bool(true_body["controls"]) and bool(false_body["controls"])
    if delta <= 0:
        if not both_controls:
            return "oppose_unassessed"
        if true_body["control_names"] != false_body["control_names"]:
            raise RuntimeError(f"control schema mismatch for source family {family}")
        control_deltas = tuple(
            float(a) - float(b) for a, b in zip(
                true_body["control_scores"], false_body["control_scores"],
                strict=True,
            )
        )
        if not all(math.isfinite(value) for value in control_deltas):
            return "oppose_unassessed"
        # Symmetric dominance: (false - true) must beat every (control_f - control_t),
        # i.e. every control delta must exceed the raw delta.
        if all(value > delta for value in control_deltas):
            return "oppose_dominant"
        return "oppose_subsignificant"
    if requires_controls and not both_controls:
        return "abstain_no_controls"
    if not both_controls:
        return "support"
    if true_body["control_names"] != false_body["control_names"]:
        raise RuntimeError(f"control schema mismatch for source family {family}")
    control_deltas = tuple(
        float(a) - float(b) for a, b in zip(
            true_body["control_scores"], false_body["control_scores"],
            strict=True,
        )
    )
    if not all(math.isfinite(value) and delta > value for value in control_deltas):
        return "abstain_positive_subdominant"
    return "support"


def classify_pair(
    stances: Mapping[str, str], registry: Mapping[str, Mapping[str, object]],
) -> str:
    supports = [name for name, s in stances.items() if s == "support"]
    dominant_opposes = [
        name for name, s in stances.items()
        if s in ("oppose_dominant", "oppose_unassessed")
    ]
    sub_opposes = [
        name for name, s in stances.items() if s == "oppose_subsignificant"
    ]
    if supports:
        # Builder-faithful blocking: an opposition blocks only when its tier
        # rank >= the strongest supporting rank.  Every family in the current
        # ledgers is tier C, so all opposition blocks; the general rule is
        # applied anyway for mixed-tier futures.
        strongest_support = max(
            int(registry[name]["tier_rank"]) for name in supports
        )
        blocking_dominant = [
            name for name in dominant_opposes
            if int(registry[name]["tier_rank"]) >= strongest_support
        ]
        blocking_subsignificant = [
            name for name in sub_opposes
            if int(registry[name]["tier_rank"]) >= strongest_support
        ]
        if blocking_dominant:
            return "contested_dominant"
        if blocking_subsignificant:
            return "vetoed_by_subsignificant_oppose"
        return "unanimous_admitted_style"
    if dominant_opposes:
        return "dominant_opposition_only"
    return "no_dominant_signal"


def molecule_rows(
    manifest: Mapping[str, np.ndarray], query: int, local_molecule: int,
) -> np.ndarray:
    # Verbatim replica of build_chemaware_dreams_native_triplets.molecule_rows.
    molecule_left, molecule_right = map(
        int, manifest["query_ptr"][query:query + 2],
    )
    count = molecule_right - molecule_left
    if not (0 <= local_molecule < count):
        raise IndexError(f"candidate {local_molecule} outside query {query}")
    molecule = molecule_left + local_molecule
    pair_left, pair_right = map(
        int, manifest["molecule_ptr"][molecule:molecule + 2],
    )
    return np.asarray(
        manifest["pair_candidate_row"][pair_left:pair_right], dtype=np.int64,
    )


class FrozenEmbeddings:
    """Mmap-backed unit-normalized row cache (replicates builder access)."""

    def __init__(self, rows_path: Path, embeddings_path: Path):
        rows = np.load(rows_path, mmap_mode="r")
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        if self.embeddings.ndim != 2 or len(rows) != len(self.embeddings):
            raise RuntimeError("frozen embedding row registry is invalid")
        if len(np.unique(rows)) != len(rows):
            raise RuntimeError("frozen embedding row registry contains duplicates")
        self.position = {int(row): index for index, row in enumerate(rows)}

    def get(self, rows: np.ndarray | list[int]) -> np.ndarray:
        try:
            positions = np.asarray(
                [self.position[int(row)] for row in rows], dtype=np.int64,
            )
        except KeyError as error:
            raise RuntimeError(
                f"reference row absent from frozen cache: {error}"
            ) from error
        values = np.asarray(self.embeddings[positions], dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(~np.isfinite(values)) or np.any(norms <= 0):
            raise RuntimeError("frozen embedding cache contains invalid vectors")
        return values / norms


def query_geometry(
    manifest: Mapping[str, np.ndarray], cache: FrozenEmbeddings, query: int,
) -> dict[str, object]:
    """Replicate candidate_geometry: molecule-max scores with anchor exclusion."""
    anchor = int(manifest["query_row"][query])
    anchor_embedding = cache.get([anchor])[0]
    left, right = map(int, manifest["query_ptr"][query:query + 2])
    labels = np.asarray(manifest["molecule_label"][left:right], dtype=bool)
    true_values = np.flatnonzero(labels)
    if len(true_values) != 1:
        raise RuntimeError(f"query {query} does not have exactly one true candidate")
    true_candidate = int(true_values[0])
    top_score: dict[int, float] = {}
    for candidate in range(right - left):
        candidate_rows = np.unique(molecule_rows(manifest, query, candidate))
        if candidate == true_candidate:
            candidate_rows = candidate_rows[candidate_rows != anchor]
        if not len(candidate_rows):
            raise RuntimeError(
                f"query {query} candidate {candidate} has no usable reference"
            )
        values = cache.get(candidate_rows) @ anchor_embedding
        top_score[candidate] = float(np.max(values))
    return {
        "anchor": anchor,
        "true_candidate": true_candidate,
        "top_score": top_score,
    }


def verify_cache(cache_dir: Path, manifest_path: Path) -> dict[str, object]:
    report_path = cache_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows_path = cache_dir / "rows.npy"
    embeddings_path = cache_dir / "embeddings_f32.npy"
    for path in (rows_path, embeddings_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = np.load(rows_path, mmap_mode="r")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    expected = {
        "manifest_sha256": file_sha256(manifest_path),
        "rows_array_sha256": array_sha256(rows),
        "embeddings_array_sha256": array_sha256(embeddings),
    }
    drift = {
        key: (report.get(key), value)
        for key, value in expected.items() if report.get(key) != value
    }
    if drift:
        raise RuntimeError(f"frozen embedding cache provenance drift: {drift}")
    if report.get("formula_role_4_accessed") is not False:
        raise RuntimeError("frozen embedding cache touched formula role 4")
    return report


def poisson_z(observed: int, expected: float) -> float:
    if expected <= 0:
        return float("nan")
    return (observed - expected) / math.sqrt(expected)


def fixed_margin_z(
    observed: int, total_errors: int, class_queries: int, total_queries: int,
) -> float:
    """Standardized residual for a 2xK table with both margins fixed."""
    if total_queries <= 1:
        return float("nan")
    expected = total_errors * class_queries / total_queries
    variance = (
        total_errors * (total_queries - total_errors)
        * class_queries * (total_queries - class_queries)
        / (total_queries * total_queries * (total_queries - 1))
    )
    if variance <= 0:
        return float("nan")
    return (observed - expected) / math.sqrt(variance)


def cluster_permutation_test(
    errors: np.ndarray, boundary_class: np.ndarray, cluster: np.ndarray,
    class_count: int, draws: int, seed: int,
) -> dict[str, object]:
    """Within-formula-cluster label permutation.

    Error labels are permuted only among queries that share a formula
    cluster, preserving cluster-level error structure; the permutation
    distribution of per-class error counts is the primary significance
    statement because query-level exchangeability is violated by clustering.
    Implemented vectorized: cluster-major ordering plus per-draw lexsort by a
    random key inside each cluster is exactly a within-cluster permutation.
    """
    if draws <= 0:
        return {"skipped": "permutation draws requested as zero"}
    rng = np.random.default_rng(seed)
    base = np.argsort(cluster, kind="stable")
    labels = errors[base].astype(bool)
    clusters_sorted = cluster[base]
    classes_sorted = boundary_class[base]
    observed = np.bincount(
        classes_sorted[labels], minlength=class_count,
    ).astype(np.int64)
    less_equal = np.zeros(class_count, dtype=np.int64)
    greater_equal = np.zeros(class_count, dtype=np.int64)
    for _ in range(draws):
        keys = rng.random(len(labels))
        order = np.lexsort((keys, clusters_sorted))
        permuted = labels[order]
        counts = np.bincount(
            classes_sorted[permuted], minlength=class_count,
        ).astype(np.int64)
        less_equal += counts <= observed
        greater_equal += counts >= observed
    names = list(BOUNDARY_CLASS_ORDER[:class_count])
    return {
        "draws": int(draws),
        "seed": int(seed),
        "clusters": int(len(np.unique(cluster))),
        "per_class": {
            name: {
                "observed": int(observed[index]),
                "p_depleted": float((less_equal[index] + 1) / (draws + 1)),
                "p_enriched": float((greater_equal[index] + 1) / (draws + 1)),
            }
            for index, name in enumerate(names)
        },
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--ledger", type=Path, action="append", required=True,
        help="Qualified candidate-source ledger directory (repeatable).",
    )
    parser.add_argument(
        "--cache-dir", type=Path, required=True,
        help="encode_chemaware_checkpoint_manifest_rows output directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-current-errors", type=int, default=None,
        help="Sanity anchor from the mining preflight (mismatch blocks decision).",
    )
    parser.add_argument(
        "--expected-admitted-winners", type=int, default=None,
        help="Sanity anchor from the mining preflight (mismatch blocks decision).",
    )
    parser.add_argument(
        "--permutation-draws", type=int, default=5000,
        help="Within-formula-cluster permutation draws (0 disables).",
    )
    parser.add_argument("--permutation-seed", type=int, default=3407)
    parser.add_argument("--maximum-detail", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.permutation_draws < 0:
        raise ValueError("--permutation-draws must be non-negative")
    manifest_path = args.manifest
    manifest = load_manifest(manifest_path)
    cache_report = verify_cache(args.cache_dir, manifest_path)
    print(
        "Frozen cache verified:",
        str(cache_report.get("checkpoint")), flush=True,
    )
    cache = FrozenEmbeddings(
        args.cache_dir / "rows.npy", args.cache_dir / "embeddings_f32.npy",
    )

    registry: dict[str, dict[str, object]] = {}
    scores: dict[int, dict[int, dict[str, dict[str, object]]]] = {}
    ledger_provenance: list[dict[str, str]] = []
    for ledger_dir in args.ledger:
        local_registry = read_family_registry(ledger_dir)
        for family, meta in local_registry.items():
            if family in registry:
                raise RuntimeError(f"family appears in two ledgers: {family}")
            registry[family] = meta
        local_scores = read_ledger_scores(ledger_dir, local_registry)
        merge_ledger_scores(scores, local_scores)
        ledger_provenance.append({
            "ledger": str(ledger_dir.resolve()),
            "report_sha256": file_sha256(ledger_dir / "report.json"),
            "tsv_sha256": file_sha256(ledger_dir / "candidate_scores.tsv"),
        })
    print(
        f"Ledgers loaded: {len(registry)} families, "
        f"{len(scores)} queries", flush=True,
    )
    validate_scores_against_manifest(scores, registry, manifest)

    pair_class_counts = {name: 0 for name in PAIR_CLASS_ORDER}
    pair_class_formula_split = {
        name: {"same_formula": 0, "cross_formula": 0} for name in PAIR_CLASS_ORDER
    }
    boundary_class_query_counts = {name: 0 for name in BOUNDARY_CLASS_ORDER}
    boundary_class_query_formula_split = {
        name: {"same_formula": 0, "cross_formula": 0}
        for name in BOUNDARY_CLASS_ORDER
    }
    error_boundary_counts = {name: 0 for name in BOUNDARY_CLASS_ORDER}
    error_boundary_formula_split = {
        name: {"same_formula": 0, "cross_formula": 0}
        for name in BOUNDARY_CLASS_ORDER
    }
    error_details: list[dict[str, object]] = []
    current_errors = 0
    tie_errors = 0
    admitted_style_error_queries = 0
    classified_queries = 0
    false_winner_ties = 0
    pairs_without_joint_family_coverage = 0
    classified_query_ids: list[int] = []
    classified_query_error_flags: list[int] = []
    classified_query_boundary_class: list[str] = []

    for query in sorted(scores):
        candidates = scores[query]
        geometry = query_geometry(manifest, cache, query)
        true_candidate = int(geometry["true_candidate"])
        if true_candidate not in candidates:
            continue
        top_score = geometry["top_score"]
        true_score = top_score[true_candidate]
        false_scores = {
            c: top_score[c] for c in top_score if c != true_candidate
        }
        if not false_scores:
            continue
        best_false_score = max(false_scores.values())
        winners = sorted(
            c for c, s in false_scores.items() if s == best_false_score
        )
        is_error = best_false_score >= true_score
        is_tie = best_false_score == true_score

        # Classify every (true, false) pair with joint family coverage.
        query_pair_classes: dict[int, str] = {}
        for false_candidate in sorted(candidates):
            if false_candidate == true_candidate:
                continue
            stances = {
                family: family_stance(
                    candidates[true_candidate][family],
                    candidates[false_candidate][family],
                    family, registry,
                )
                for family in sorted(registry)
                if family in candidates[true_candidate]
                and family in candidates[false_candidate]
            }
            if not stances:
                pairs_without_joint_family_coverage += 1
                continue
            pair_class = classify_pair(stances, registry)
            query_pair_classes[false_candidate] = pair_class
            same_formula = (
                str(manifest["molecule_formula"][
                    int(manifest["query_ptr"][query]) + true_candidate
                ])
                == str(manifest["molecule_formula"][
                    int(manifest["query_ptr"][query]) + false_candidate
                ])
            )
            pair_class_counts[pair_class] += 1
            pair_class_formula_split[pair_class][
                "same_formula" if same_formula else "cross_formula"
            ] += 1
        if not query_pair_classes:
            # No classified pair at all: the query carries no admissible
            # boundary information, but its winning-false boundary is still
            # attributable to absent coverage rather than weak evidence.
            continue
        classified_queries += 1
        classified_query_ids.append(int(query))
        false_winner_ties += int(len(winners) > 1)

        # Boundary attribution across ALL tied winners; never trust an
        # arbitrary tie-break, and never conflate absent coverage with weak
        # evidence.
        winner_classes: dict[int, str] = {}
        for winner in winners:
            if winner in query_pair_classes:
                winner_classes[winner] = query_pair_classes[winner]
            else:
                winner_classes[winner] = "no_ledger_coverage"
        distinct = sorted(set(winner_classes.values()))
        boundary_class = distinct[0] if len(distinct) == 1 else "ambiguous_tie"
        boundary_same_formula = (
            str(manifest["molecule_formula"][
                int(manifest["query_ptr"][query]) + true_candidate
            ])
            == str(manifest["molecule_formula"][
                int(manifest["query_ptr"][query]) + winners[0]
            ])
        )
        boundary_class_query_counts[boundary_class] += 1
        boundary_class_query_formula_split[boundary_class][
            "same_formula" if boundary_same_formula else "cross_formula"
        ] += 1
        classified_query_error_flags.append(int(bool(is_error)))
        classified_query_boundary_class.append(boundary_class)

        if is_error:
            current_errors += 1
            if is_tie:
                tie_errors += 1
            error_boundary_counts[boundary_class] += 1
            error_boundary_formula_split[boundary_class][
                "same_formula" if boundary_same_formula else "cross_formula"
            ] += 1
            if boundary_class == "unanimous_admitted_style":
                admitted_style_error_queries += 1
            if len(error_details) < args.maximum_detail:
                error_details.append({
                    "query": int(query),
                    "true_candidate": int(true_candidate),
                    "winning_false_candidates": [int(w) for w in winners],
                    "boundary_tie": bool(is_tie),
                    "boundary_class": boundary_class,
                    "boundary_classes_per_winner": {
                        str(w): winner_classes[w] for w in winners
                    },
                    "true_score": float(true_score),
                    "best_false_score": float(best_false_score),
                    "query_pair_classes": {
                        str(c): cls
                        for c, cls in sorted(query_pair_classes.items())
                    },
                })

    total_pairs = sum(pair_class_counts.values())
    uniform_expectation = {
        name: (
            current_errors * boundary_class_query_counts[name] / classified_queries
            if classified_queries else 0.0
        )
        for name in BOUNDARY_CLASS_ORDER
    }
    depletion = {
        name: {
            "queries_with_this_top_false_boundary": boundary_class_query_counts[name],
            "descriptive_pair_count": (
                pair_class_counts[name] if name in PAIR_CLASS_ORDER else None
            ),
            "error_boundaries": error_boundary_counts[name],
            "uniform_expected": uniform_expectation[name],
            "poisson_z": poisson_z(
                error_boundary_counts[name], uniform_expectation[name],
            ),
            "fixed_margin_z": fixed_margin_z(
                error_boundary_counts[name], current_errors,
                boundary_class_query_counts[name], classified_queries,
            ),
        }
        for name in BOUNDARY_CLASS_ORDER
    }
    permutation = cluster_permutation_test(
        np.asarray(classified_query_error_flags, dtype=bool),
        np.asarray(
            [BOUNDARY_CLASS_ORDER.index(c) for c in classified_query_boundary_class],
            dtype=np.int64,
        ),
        np.asarray(
            [str(manifest["query_formula"][q]) for q in classified_query_ids],
            dtype=object,
        ),
        len(BOUNDARY_CLASS_ORDER),
        args.permutation_draws,
        args.permutation_seed,
    )

    teachable_now = error_boundary_counts["unanimous_admitted_style"]
    recoverable = error_boundary_counts["vetoed_by_subsignificant_oppose"]
    contested = error_boundary_counts["contested_dominant"]
    dead = (
        error_boundary_counts["dominant_opposition_only"]
        + error_boundary_counts["no_dominant_signal"]
    )

    anchor_checks = {}
    for anchor_name, observed, expected_value in (
        ("current_errors", current_errors, args.expected_current_errors),
        ("admitted_style_winners", admitted_style_error_queries,
         args.expected_admitted_winners),
    ):
        matches = expected_value is None or observed == expected_value
        anchor_checks[anchor_name] = {
            "observed": observed, "expected": expected_value, "matches": matches,
        }
        if not matches:
            print(
                f"  WARNING: {anchor_name}={observed} differs from preflight "
                f"anchor {expected_value}; geometry or ledger drift suspected",
                file=sys.stderr, flush=True,
            )

    anchors_supplied = (
        args.expected_current_errors is not None
        and args.expected_admitted_winners is not None
    )
    anchors_match = all(body["matches"] for body in anchor_checks.values())
    output = {
        "status": STATUS,
        "provenance": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
            "ledgers": ledger_provenance,
            "embedding_cache": {
                "report": str((args.cache_dir / "report.json").resolve()),
                "checkpoint": cache_report.get("checkpoint"),
                "checkpoint_sha256": cache_report.get("checkpoint_sha256"),
                "checkpoint_kind": cache_report.get("checkpoint_kind"),
                "rows": int(cache_report.get("rows", 0)),
                "queries_in_cache_report": cache_report.get(
                    "formula_role_query_counts", {}
                ),
            },
        },
        "families": registry,
        "class_definitions": CLASS_DEFINITIONS,
        "caveats": list(CAVEATS),
        "classified_queries": classified_queries,
        "total_pairs": total_pairs,
        "pairs_without_joint_family_coverage": pairs_without_joint_family_coverage,
        "pair_class_counts": pair_class_counts,
        "pair_class_formula_split": pair_class_formula_split,
        "top_false_boundary_class_query_counts": boundary_class_query_counts,
        "top_false_boundary_class_formula_split": boundary_class_query_formula_split,
        "current_errors": current_errors,
        "tie_errors": tie_errors,
        "multiple_top_false_ties": false_winner_ties,
        "error_boundary_counts": error_boundary_counts,
        "error_boundary_formula_split": error_boundary_formula_split,
        "uniform_baseline": depletion,
        "cluster_permutation_test": permutation,
        "sanity_anchors": {
            "current_errors": current_errors,
            "admitted_style_error_boundaries": admitted_style_error_queries,
            "expected_current_errors": args.expected_current_errors,
            "expected_admitted_winners": args.expected_admitted_winners,
            "checks": anchor_checks,
            "all_required_anchors_supplied": anchors_supplied,
            "all_match": anchors_match,
        },
        "decision_summary": {
            "teachable_under_current_rules": teachable_now,
            "recoverable_by_symmetric_significance": recoverable,
            "contested_dominant": contested,
            "no_chemical_direction_available": dead,
            "boundary_without_ledger_coverage":
                error_boundary_counts["no_ledger_coverage"],
            "ambiguous_tie_boundaries": error_boundary_counts["ambiguous_tie"],
            "decision_eligible": anchors_supplied and anchors_match,
            "arithmetic": (
                "Counts are train-panel opportunities, not predicted "
                "selection-panel corrections. Winner-to-percentage conversion "
                "requires the panel size from the executed role-2 evaluation "
                "report (queries field); no hardcoded panel size is assumed "
                "here. Performance claims require an executed formula-disjoint "
                "role-2 evaluation."
            ),
        },
        "error_boundary_details": error_details,
        "formula_role_4_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Evidence-class diagnostic complete -> {args.output}", flush=True)
    print(json.dumps({
        "current_errors": current_errors,
        "error_boundary_counts": error_boundary_counts,
        "pair_class_counts": pair_class_counts,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
