"""Pure helpers for E15 source-local calibration and balanced panels."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


def robust_location_scale(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("cannot calibrate an empty/non-finite group")
    center = float(np.median(array))
    mad = float(np.median(np.abs(array - center))) * 1.4826
    q25, q75 = np.quantile(array, [0.25, 0.75])
    iqr_scale = float(q75 - q25) / 1.349
    scale = max(mad, iqr_scale, 1e-6)
    return center, scale


def calibrate_source_local(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate action strength without comparing raw margins across geometries."""
    required = {"source", "action_family", "supervision_kind", "margin_delta"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"calibration frame misses {sorted(missing)}")
    output = frame.copy()
    kind = output["supervision_kind"].astype(str)
    if not kind.isin({"corrective", "harmful"}).all():
        raise ValueError("calibration accepts only corrective/harmful rows")
    delta = output["margin_delta"].to_numpy(np.float64)
    output["directional_strength"] = np.where(kind.eq("corrective"), delta, -delta)
    if (output["directional_strength"] <= 0).any():
        bad = output.loc[output["directional_strength"] <= 0,
                         ["source", "action_family", "supervision_kind", "margin_delta"]]
        raise RuntimeError(f"rank-changing actions have non-directional margins: {bad.head().to_dict('records')}")

    records: list[dict[str, object]] = []
    output["calibration_group"] = ""
    output["calibrated_strength"] = np.nan
    output["source_kind_percentile"] = np.nan
    for (source, supervision), source_block in output.groupby(
        ["source", "supervision_kind"], sort=True,
    ):
        source_values = source_block["directional_strength"].rank(
            method="average", pct=True,
        )
        output.loc[source_block.index, "source_kind_percentile"] = source_values
        family_counts = source_block["action_family"].value_counts()
        for family, family_block in source_block.groupby("action_family", sort=True):
            # Small families borrow scale only from their own source/kind.  They never
            # borrow from another geometry or supervision branch.
            if int(family_counts[family]) >= 20:
                fit = family_block
                level = "source_kind_family"
            else:
                fit = source_block
                level = "source_kind_fallback"
            center, scale = robust_location_scale(fit["directional_strength"])
            group = f"{source}|{supervision}|{family}|{level}"
            z = (family_block["directional_strength"].to_numpy(np.float64) - center) / scale
            output.loc[family_block.index, "calibration_group"] = group
            output.loc[family_block.index, "calibrated_strength"] = np.clip(z, -3.0, 3.0)
            records.append({
                "source": str(source), "supervision_kind": str(supervision),
                "action_family": str(family), "level": level,
                "n_group": int(len(family_block)), "n_fit": int(len(fit)),
                "center": center, "scale": scale,
            })
    if output[["calibrated_strength", "source_kind_percentile"]].isna().any().any():
        raise RuntimeError("source-local calibration left missing values")
    return output, pd.DataFrame(records)


def diverse_panel(frame: pd.DataFrame, per_source_kind: int, seed: int) -> pd.DataFrame:
    """Select a finite, non-recycled panel balanced by source and branch."""
    if per_source_kind < 1:
        raise ValueError("per_source_kind must be positive")
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    for key, block in frame.groupby(["source", "supervision_kind"], sort=True):
        if len(block) < per_source_kind:
            raise RuntimeError(f"insufficient unique actions for panel stratum {key}: {len(block)}")
        block = block.copy()
        block["_jitter"] = rng.random(len(block))
        block = block.sort_values(
            ["query_formula", "query_ik14", "source_kind_percentile", "_jitter", "action_id"],
            ascending=[True, True, False, True, True], kind="stable",
        )
        # First pass maximizes identity diversity; second pass fills the exact quota.
        first = block.drop_duplicates("query_ik14", keep="first").head(per_source_kind)
        if len(first) < per_source_kind:
            first = pd.concat([
                first, block.loc[~block.index.isin(first.index)].head(per_source_kind - len(first)),
            ])
        first = first.drop(columns="_jitter")
        first["panel_stratum"] = f"{key[0]}|{key[1]}"
        selected.append(first)
    panel = pd.concat(selected, ignore_index=True)
    keys = ["source", "query_index", "action_id", "supervision_kind"]
    if panel.duplicated(keys).any():
        raise RuntimeError("calibration panel repeats an action")
    observed = panel.groupby(["source", "supervision_kind"]).size()
    if not (observed == per_source_kind).all():
        raise RuntimeError("calibration panel is not exactly balanced")
    return panel


def inverse_source_weights(frame: pd.DataFrame) -> dict[str, float]:
    """Equalize total source mass while preserving every unique action once."""
    counts = frame.groupby(["supervision_kind", "source"]).size()
    sources_by_kind: dict[str, list[str]] = defaultdict(list)
    for supervision, source in counts.index:
        sources_by_kind[str(supervision)].append(str(source))
    weights: dict[str, float] = {}
    for (supervision, source), count in counts.items():
        n_sources = len(sources_by_kind[str(supervision)])
        weights[f"{supervision}|{source}"] = 1.0 / (n_sources * int(count))
    return weights
