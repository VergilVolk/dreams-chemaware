#!/usr/bin/env python
"""Diagnose conservative RT correction for MTBLS13729 OpenMS features.

This pilot deliberately separates model fitting from evaluation. Candidate
landmarks are mutual nearest-m/z features within a broad RT window. Half are
used to estimate a shift/robust linear map and half are held out. A correction
is acceptable only when held-out RT residuals and ordinary feature matching do
not deteriorate relative to leaving RT unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_mtbls13729_openms_features import greedy_match


def mutual_landmarks(a: pd.DataFrame, b: pd.DataFrame, ppm: float, broad_rt: float) -> pd.DataFrame:
    """Return one-to-one, mutual nearest-m/z landmarks with a broad RT guard."""
    aa = a.reset_index(drop=True)
    bb = b.reset_index(drop=True)
    a_mz = aa["mz"].to_numpy(float)
    b_mz = bb["mz"].to_numpy(float)
    a_rt = aa["rt_sec"].to_numpy(float)
    b_rt = bb["rt_sec"].to_numpy(float)

    def nearest(src_mz: np.ndarray, src_rt: np.ndarray, dst_mz: np.ndarray, dst_rt: np.ndarray) -> np.ndarray:
        out = np.full(len(src_mz), -1, dtype=int)
        order = np.argsort(dst_mz)
        dmz = dst_mz[order]
        for i, (mz, rt) in enumerate(zip(src_mz, src_rt)):
            tol = mz * ppm * 1e-6
            lo = int(np.searchsorted(dmz, mz - tol, side="left"))
            hi = int(np.searchsorted(dmz, mz + tol, side="right"))
            if lo == hi:
                continue
            candidates = order[lo:hi]
            rt_ok = candidates[np.abs(dst_rt[candidates] - rt) <= broad_rt]
            if not len(rt_ok):
                continue
            delta_ppm = np.abs(dst_mz[rt_ok] - mz) / mz * 1e6
            # m/z is the primary identity key; RT only resolves near ties.
            cost = delta_ppm + 0.01 * np.abs(dst_rt[rt_ok] - rt) / broad_rt
            out[i] = int(rt_ok[int(np.argmin(cost))])
        return out

    ab = nearest(a_mz, a_rt, b_mz, b_rt)
    ba = nearest(b_mz, b_rt, a_mz, a_rt)
    rows = []
    for i, j in enumerate(ab):
        if j >= 0 and ba[j] == i:
            key = f"{a_mz[i]:.6f}|{a_rt[i]:.3f}|{b_mz[j]:.6f}|{b_rt[j]:.3f}"
            split = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2
            rows.append(
                {
                    "a_index": i,
                    "b_index": j,
                    "a_mz": a_mz[i],
                    "b_mz": b_mz[j],
                    "a_rt": a_rt[i],
                    "b_rt": b_rt[j],
                    "delta_rt": a_rt[i] - b_rt[j],
                    "split": "train" if split == 0 else "test",
                    "weight": math.sqrt(max(float(aa.loc[i, "intensity"]), 1.0) * max(float(bb.loc[j, "intensity"]), 1.0)),
                }
            )
    return pd.DataFrame(rows)


def mad_clip(values: np.ndarray, z: float = 4.0) -> np.ndarray:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad <= 1e-12:
        return np.ones(len(values), dtype=bool)
    return np.abs(values - med) <= z * 1.4826 * mad


def fit_models(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Models map b RT to the a/reference RT: corrected = intercept + slope*b."""
    delta = train["delta_rt"].to_numpy(float)
    keep = mad_clip(delta)
    clipped = train.loc[keep]
    shift = float(np.median(clipped["delta_rt"]))

    x = clipped["b_rt"].to_numpy(float)
    y = clipped["a_rt"].to_numpy(float)
    # Robust Theil-Sen from a deterministic subset to keep CPU cost bounded.
    if len(x) > 400:
        idx = np.linspace(0, len(x) - 1, 400).astype(int)
        x, y = x[idx], y[idx]
    slopes = []
    for i in range(len(x)):
        dx = x[i + 1 :] - x[i]
        valid = np.abs(dx) > 1e-9
        if np.any(valid):
            slopes.extend(((y[i + 1 :][valid] - y[i]) / dx[valid]).tolist())
    slope = float(np.median(slopes)) if slopes else 1.0
    intercept = float(np.median(y - slope * x)) if len(x) else shift
    return {"none": (0.0, 1.0), "median_shift": (shift, 1.0), "theil_sen": (intercept, slope)}


def corrected(frame: pd.DataFrame, model: tuple[float, float]) -> pd.DataFrame:
    out = frame.copy()
    intercept, slope = model
    out["rt_sec"] = intercept + slope * out["rt_sec"].to_numpy(float)
    return out


def evaluate_landmarks(test: pd.DataFrame, model: tuple[float, float]) -> dict[str, float]:
    intercept, slope = model
    residual = test["a_rt"].to_numpy(float) - (intercept + slope * test["b_rt"].to_numpy(float))
    return {
        "n": int(len(residual)),
        "median_abs_error_sec": float(np.median(np.abs(residual))),
        "p90_abs_error_sec": float(np.percentile(np.abs(residual), 90)),
        "within_5_sec": float(np.mean(np.abs(residual) <= 5)),
        "within_10_sec": float(np.mean(np.abs(residual) <= 10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/mtbls13729/ms1_feature_pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/ms1_rt_diagnosis"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--samples", nargs="+", default=["P01-Ltu", "P01-LN", "P21-Rmu", "P21-RN"])
    parser.add_argument("--noise-threshold", type=float, default=10000.0)
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--broad-rt-sec", type=float, default=60.0)
    parser.add_argument("--rt-tolerances", nargs="+", type=float, default=[5, 10, 15, 20, 30])
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"status": "complete", "panels": {}}
    pairs = ((0, 1), (2, 3))
    for panel in args.panels:
        frames = []
        for sample in args.samples:
            path = args.pilot_dir / "features" / f"{panel}__{sample}__noise_{args.noise_threshold:g}.csv.gz"
            frames.append(pd.read_csv(path))
        panel_result = {}
        for ai, bi in pairs:
            a, b = frames[ai], frames[bi]
            pair_name = f"{args.samples[ai]}__{args.samples[bi]}"
            landmarks = mutual_landmarks(a, b, args.ppm, args.broad_rt_sec)
            landmarks.to_csv(out / f"{panel}__{pair_name}__landmarks.csv.gz", index=False)
            train = landmarks[landmarks["split"] == "train"]
            test = landmarks[landmarks["split"] == "test"]
            models = fit_models(train)
            model_result = {}
            for name, model in models.items():
                b_corr = corrected(b, model)
                matches = {str(rt): greedy_match(a, b_corr, args.ppm, rt) for rt in args.rt_tolerances}
                model_result[name] = {
                    "intercept_sec": model[0],
                    "slope": model[1],
                    "heldout_landmarks": evaluate_landmarks(test, model),
                    "all_feature_matching": matches,
                }
            panel_result[pair_name] = {
                "n_features_a": len(a),
                "n_features_b": len(b),
                "n_landmarks": len(landmarks),
                "train_landmarks": len(train),
                "test_landmarks": len(test),
                "raw_delta_rt_median": float(landmarks["delta_rt"].median()),
                "raw_delta_rt_mad": float(np.median(np.abs(landmarks["delta_rt"] - landmarks["delta_rt"].median()))),
                "models": model_result,
            }
        report["panels"][panel] = panel_result

    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
