from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASE = Path("data/mtbls13729/biology_closure_analysis_v1")
OUTPUT = Path("data/mtbls13729/biology_axes_analysis_v1")
FEATURES = {
    4966: "purine_like_C7H9N5O",
    1717: "diacetylspermidine_like",
    3222: "C20_4_acylcarnitine_like",
}
FILES = {
    "raw": "rmu_raw_pair_deltas.csv",
    "global_pqn_prev60": "rmu_global_pqn_prev60_pair_deltas.csv",
    "global_pqn_prev80": "rmu_global_pqn_prev80_pair_deltas.csv",
    "global_pqn_prev90": "rmu_global_pqn_prev90_pair_deltas.csv",
}


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int = 20260830) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(10_000):
        index = rng.integers(0, len(x), len(x))
        if np.unique(x[index]).size < 2 or np.unique(y[index]).size < 2:
            continue
        values.append(float(stats.spearmanr(x[index], y[index]).statistic))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    module = pd.read_csv(BASE / "fully_ion_family_collapsed_module_patient_effects.csv")
    report: dict[str, object] = {
        "status": "mtbls13729_biology_axes_analysis_complete",
        "primary_axis": "modified_guanosine_module",
        "correlations": {},
    }
    rows = []
    for normalization, filename in FILES.items():
        feature = pd.read_csv(BASE / filename).pivot(index="patient", columns="feature_id", values="delta_log2")
        module_effect = module[
            (module["normalization"] == normalization) & (module["cohort"] == "Rmu")
        ].set_index("patient")["module_log2fc"]
        normalization_report = {}
        for feature_id, name in FEATURES.items():
            joined = pd.concat([module_effect.rename("module"), feature[feature_id].rename("axis")], axis=1).dropna()
            spearman = stats.spearmanr(joined["module"], joined["axis"])
            pearson = stats.pearsonr(joined["module"], joined["axis"])
            item = {
                "feature_id": feature_id,
                "n": int(len(joined)),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "spearman_patient_bootstrap_95ci": bootstrap_spearman(
                    joined["module"].to_numpy(), joined["axis"].to_numpy(), seed=20260830 + feature_id
                ),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
            }
            normalization_report[name] = item
            rows.append({"normalization": normalization, "axis": name, **item})
        report["correlations"][normalization] = normalization_report

    raw = pd.read_csv(BASE / FILES["raw"]).pivot(index="patient", columns="feature_id", values="delta_log2")
    raw_module = module[(module["normalization"] == "raw") & (module["cohort"] == "Rmu")].set_index("patient")[
        "module_log2fc"
    ]
    joint = pd.concat(
        [raw_module.rename("modified_guanosine_module"), raw[list(FEATURES)].rename(columns=FEATURES)], axis=1
    )
    report["raw_directional_signature"] = {
        column: {
            "available": int(joint[column].notna().sum()),
            "positive": int((joint[column] > 0).sum()),
        }
        for column in joint.columns
    }
    report["interpretation"] = (
        "The purine-like feature is an independent feature outside the collapsed modified-guanosine module. "
        "Correlation is patient-level co-variation, not evidence that either identity or pathway causes the other."
    )

    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUTPUT / "correlations.csv", index=False)
    joint.reset_index().to_csv(OUTPUT / "raw_patient_axes.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
