"""Pure, testable contracts for E15 multi-action noise fine-tuning."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ExposureItem:
    query_index: int
    identity: str
    formula: str
    source: str
    family: str
    action_id: str
    supervision_kind: str


def bounded_stratified_epoch(
    examples: Sequence[ExposureItem], rng: np.random.Generator,
    maximum_exposure: int = 1,
) -> tuple[list[ExposureItem], dict[str, object]]:
    """Return one finite epoch; never wrap a cursor or silently recycle rows."""
    if maximum_exposure < 1:
        raise ValueError("maximum exposure must be positive")
    if not examples:
        return [], {
            "unique_actions": 0, "draws": 0, "maximum_exposure": 0,
            "p50_exposure": 0.0, "p90_exposure": 0.0,
        }
    unique: dict[tuple[int, str, str], ExposureItem] = {}
    for example in examples:
        key = (int(example.query_index), str(example.action_id), str(example.supervision_kind))
        if key in unique:
            raise RuntimeError(f"duplicate E15 exposure key: {key}")
        unique[key] = example

    strata: dict[tuple[str, str, str], list[ExposureItem]] = defaultdict(list)
    for example in unique.values():
        strata[(example.supervision_kind, example.source, example.family)].append(example)
    for values in strata.values():
        values.sort(key=lambda value: (value.formula, value.identity, value.query_index, value.action_id))
        rng.shuffle(values)

    ordered: list[ExposureItem] = []
    keys = sorted(strata)
    cursors = Counter()
    while True:
        progressed = False
        for key in keys:
            cursor = cursors[key]
            values = strata[key]
            if cursor < len(values):
                ordered.append(values[cursor])
                cursors[key] += 1
                progressed = True
        if not progressed:
            break

    # Explicit repeats, if ever authorized, are whole-ledger repeats rather
    # than hidden cursor wrap. The exposure count remains auditable.
    epoch = [item for _ in range(maximum_exposure) for item in ordered]
    rng.shuffle(epoch)
    exposure = Counter((item.query_index, item.action_id, item.supervision_kind) for item in epoch)
    values = np.asarray(list(exposure.values()), dtype=np.int64)
    report = {
        "unique_actions": int(len(exposure)),
        "draws": int(len(epoch)),
        "maximum_exposure": int(values.max()),
        "p50_exposure": float(np.quantile(values, 0.50)),
        "p90_exposure": float(np.quantile(values, 0.90)),
        "strata": int(len(strata)),
    }
    if report["maximum_exposure"] > maximum_exposure:
        raise RuntimeError("bounded E15 sampler exceeded its exposure contract")
    return epoch, report


def corrective_margin_loss(
    clean_margin: torch.Tensor,
    action_margin: torch.Tensor,
    baseline_margin: torch.Tensor,
    observed_delta: torch.Tensor,
    *,
    rank_margin: float = 0.05,
    temperature: float = 0.10,
    delta_fraction: float = 0.50,
    delta_cap: float = 0.20,
) -> torch.Tensor:
    """Positive teacher: transfer only a conservative fraction of action gain."""
    target = baseline_margin + delta_fraction * torch.clamp(
        observed_delta, min=0.0, max=delta_cap,
    )
    clean_rank = F.softplus((rank_margin - clean_margin) / temperature)
    action_rank = F.softplus((rank_margin - action_margin) / temperature)
    transfer = F.relu(target - clean_margin)
    return clean_rank.mean() + action_rank.mean() + transfer.mean()


def risk_protection_loss(
    clean_margin: torch.Tensor,
    baseline_margin: torch.Tensor,
    clean_preservation: torch.Tensor,
    *,
    floor_slack: float = 0.005,
    preservation_floor: float = 0.995,
) -> torch.Tensor:
    """Negative control: protect clean geometry; never imitate a harmful action."""
    margin_floor = F.relu(baseline_margin - floor_slack - clean_margin)
    preservation = F.relu(preservation_floor - clean_preservation)
    return margin_floor.mean() + preservation.mean()


def project_corrective_against_risk(
    corrective: Sequence[torch.Tensor | None],
    risk: Sequence[torch.Tensor | None],
    *,
    minimum_risk_norm: float = 1e-6,
) -> tuple[list[torch.Tensor | None], dict[str, float | bool]]:
    """PCGrad-style veto: remove only the component conflicting with risk."""
    dot = torch.zeros((), dtype=torch.float64)
    risk_sq = torch.zeros((), dtype=torch.float64)
    corr_sq = torch.zeros((), dtype=torch.float64)
    for left, right in zip(corrective, risk):
        if left is not None:
            corr_sq += torch.sum(left.detach().double() ** 2).cpu()
        if right is not None:
            risk_sq += torch.sum(right.detach().double() ** 2).cpu()
        if left is not None and right is not None:
            dot += torch.sum(left.detach().double() * right.detach().double()).cpu()
    risk_norm = float(torch.sqrt(risk_sq))
    corrective_norm = float(torch.sqrt(corr_sq))
    risk_active = bool(risk_norm >= minimum_risk_norm)
    conflict = bool(dot < 0 and risk_active)
    scale = float(dot / risk_sq) if conflict else 0.0
    output: list[torch.Tensor | None] = []
    for left, right in zip(corrective, risk):
        if left is None:
            output.append(None)
        elif conflict and right is not None:
            output.append(left - scale * right.to(left.device, left.dtype))
        else:
            output.append(left.clone())
    denominator = float(torch.sqrt(corr_sq * risk_sq))
    return output, {
        "gradient_dot": float(dot),
        "gradient_cosine": float(dot) / denominator if denominator > 0 else float("nan"),
        "corrective_gradient_norm": corrective_norm,
        "risk_gradient_norm": risk_norm,
        "risk_projection_active": risk_active,
        "conflict": conflict,
        "projection_scale": scale,
    }


def stratified_calibration_indices(
    examples: Sequence[ExposureItem], rng: np.random.Generator,
    *, microbatches: int = 32, batch_size: int = 4,
) -> list[list[int]]:
    """At least 32 stratified batches and 128 observations, never ``[:4]``."""
    if microbatches < 32 or microbatches * batch_size < 128:
        raise ValueError("E15 calibration requires >=32 batches and >=128 observations")
    if len(examples) < microbatches * batch_size:
        raise RuntimeError("insufficient unique observations for E15 gradient calibration")
    by_stratum: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_stratum[(example.supervision_kind, example.source, example.family)].append(index)
    for values in by_stratum.values():
        rng.shuffle(values)
    keys = sorted(by_stratum)
    cursor = Counter()
    selected: list[int] = []
    while len(selected) < microbatches * batch_size:
        progressed = False
        for key in keys:
            values = by_stratum[key]
            position = cursor[key]
            if position < len(values):
                selected.append(values[position])
                cursor[key] += 1
                progressed = True
                if len(selected) == microbatches * batch_size:
                    break
        if not progressed:
            raise RuntimeError("stratified calibration exhausted unique observations")
    if len(set(selected)) != len(selected):
        raise RuntimeError("E15 calibration repeated an observation")
    return [selected[left:left + batch_size] for left in range(0, len(selected), batch_size)]


def exposure_items(records: Iterable[dict[str, object]]) -> list[ExposureItem]:
    return [ExposureItem(
        query_index=int(row["query_index"]),
        identity=str(row["query_ik14"]),
        formula=str(row["query_formula"]),
        source=str(row["source"]),
        family=str(row["action_family"]),
        action_id=str(row["action_id"]),
        supervision_kind=str(row["supervision_kind"]),
    ) for row in records]
