"""Formula- and polarity-aware transforms for ChemAware rule responses."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


@dataclass(frozen=True)
class RuleChannel:
    category: str
    mode: str
    formula: str | None
    name: str
    value: float


def parse_formula(formula: str) -> dict[str, int] | None:
    formula = str(formula).strip()
    # The manifest has one conventional terminal charge annotation
    # (e.g. C8H18OP+).  Charge is not an element count and is removed only when
    # it is a terminal sign; internal punctuation remains fail-closed.
    formula = formula.rstrip("+-")
    if not formula:
        return None
    tokens = FORMULA_TOKEN.findall(formula)
    rebuilt = "".join(element + count for element, count in tokens)
    if not tokens or rebuilt != formula:
        return None
    result: dict[str, int] = {}
    for element, count in tokens:
        result[element] = result.get(element, 0) + (int(count) if count else 1)
    return result


def formula_from_rule(record: dict[str, object]) -> str | None:
    explicit = str(record.get("formula", "")).strip()
    if parse_formula(explicit) is not None:
        return explicit
    suffix = str(record.get("name", "")).split(":", 1)[-1]
    return suffix if parse_formula(suffix) is not None else None


def registry(records: list[dict[str, object]]) -> list[RuleChannel]:
    selected = [
        record for record in records
        if (
            record.get("category") == "NL" and record.get("match_type") == "mass_diff"
        ) or (
            record.get("category") == "CF" and record.get("match_type") == "peak_mz"
        )
    ]
    return [
        RuleChannel(
            category=str(record["category"]),
            mode=str(record.get("mode", "unspecified")),
            formula=formula_from_rule(record),
            name=str(record.get("name", "")),
            value=float(record.get("value_da", record.get("value"))),
        )
        for record in selected
    ]


def positive_mode_mask(channels: list[RuleChannel]) -> np.ndarray:
    return np.asarray([channel.mode != "neg" for channel in channels], dtype=bool)


def is_subformula(fragment: dict[str, int], precursor: dict[str, int]) -> bool:
    return all(count <= precursor.get(element, 0) for element, count in fragment.items())


def formula_gate(
    channels: list[RuleChannel], precursor_formula: str,
    assigned_formula: list[str | None] | None = None,
) -> np.ndarray:
    precursor = parse_formula(precursor_formula)
    if precursor is None:
        raise ValueError(f"cannot parse precursor formula: {precursor_formula!r}")
    formulas = assigned_formula if assigned_formula is not None else [x.formula for x in channels]
    if len(formulas) != len(channels):
        raise ValueError("assigned formula registry is misaligned")
    output = positive_mode_mask(channels)
    for index, formula in enumerate(formulas):
        parsed = parse_formula(formula) if formula is not None else None
        if parsed is not None:
            output[index] &= is_subformula(parsed, precursor)
    return output


def shuffled_formula_assignment(
    channels: list[RuleChannel], seed: int,
) -> list[str | None]:
    """Break rule-mass/formula coupling while preserving category and gate density."""
    assigned = [channel.formula for channel in channels]
    rng = np.random.default_rng(seed)
    for category in sorted({channel.category for channel in channels}):
        index = [
            i for i, channel in enumerate(channels)
            if channel.category == category and channel.mode != "neg" and channel.formula is not None
        ]
        values = [assigned[i] for i in index]
        permutation = rng.permutation(len(values))
        for target, source in zip(index, permutation, strict=True):
            assigned[target] = values[int(source)]
    return assigned


def inverse_document_frequency(
    responses: np.ndarray, allowed: np.ndarray,
) -> np.ndarray:
    responses = np.asarray(responses)
    allowed = np.asarray(allowed, dtype=bool)
    if responses.ndim != 2 or responses.shape[1] != len(allowed):
        raise ValueError("response matrix and rule registry are misaligned")
    document_frequency = np.sum(responses > 0, axis=0)
    value = np.log((len(responses) + 1.0) / (document_frequency + 1.0)) + 1.0
    value[~allowed] = 0.0
    return value.astype(np.float32)


def normalized_formula_response(
    response: np.ndarray, idf: np.ndarray, gate: np.ndarray,
) -> np.ndarray:
    value = np.asarray(response, dtype=np.float32) * np.asarray(idf, dtype=np.float32) * np.asarray(gate, dtype=np.float32)
    if value.ndim == 1:
        norm = float(np.linalg.norm(value))
        return value / max(norm, 1e-12)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norm, 1e-12)


def rows_for_queries(queries: np.ndarray, body: dict[str, np.ndarray]) -> np.ndarray:
    rows: list[int] = []
    for query in map(int, queries):
        rows.append(int(body["query_row"][query]))
        left, right = map(int, body["query_ptr"][query:query + 2])
        ref_left = int(body["molecule_ptr"][left])
        ref_right = int(body["molecule_ptr"][right])
        rows.extend(map(int, body["pair_candidate_row"][ref_left:ref_right]))
    return np.unique(np.asarray(rows, dtype=np.int64))


def append_formula_rule_scores(
    scored: dict[str, np.ndarray], queries: np.ndarray, body: dict[str, np.ndarray],
    cache, idf: np.ndarray, channels: list[RuleChannel],
    assigned_formula: list[str | None] | None, output_key: str,
) -> None:
    output = np.empty(len(queries), dtype=object)
    gate_cache: dict[str, np.ndarray] = {}
    for index, query in enumerate(map(int, queries)):
        formula = str(body["query_formula"][query])
        gate = gate_cache.setdefault(
            formula, formula_gate(channels, formula, assigned_formula),
        )
        qrow = int(body["query_row"][query])
        left, right = map(int, body["query_ptr"][query:query + 2])
        ref_left = int(body["molecule_ptr"][left])
        ref_right = int(body["molecule_ptr"][right])
        reference_rows = body["pair_candidate_row"][ref_left:ref_right]
        query_response = normalized_formula_response(
            cache.get(qrow)["rule_response"], idf, gate,
        )
        reference_response = normalized_formula_response(
            np.stack([cache.get(int(row))["rule_response"] for row in reference_rows]),
            idf, gate,
        )
        output[index] = reference_response @ query_response
    scored[output_key] = output
