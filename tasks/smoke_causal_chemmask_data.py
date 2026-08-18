"""Fast data-layer smoke test for causal ChemMask sampling."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from train_causal_chemmask_head import CausalDynamicTripletDataset
from train_e1_identity import CandidatePool


ROOT = Path(__file__).resolve().parent.parent


def decode(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def main() -> None:
    data_path = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
    pool = CandidatePool(ROOT / "data/e1/e1_train_triplet_pool_10ppm.npz")
    dataset = CausalDynamicTripletDataset(
        data_path=data_path,
        pool=pool,
        n_highest_peaks=100,
        seed=20260815,
        hard_negative_prob=1.0,
        negative_probe_size=8,
        identity_mask_prob=1.0,
        identity_mask_max_fraction=0.3,
        identity_mask_max_peaks=12,
        fragment_tolerance=0.02,
        length=128,
        fixed=True,
    )
    rows = [dataset[index] for index in range(len(dataset))]
    with h5py.File(data_path, "r") as handle:
        masses = np.asarray(handle["precursor_mz"])
        ik = handle["INCHIKEY"]
        folds = handle["fold"]
        protocol_errors = []
        ppm_values = []
        for row in rows:
            anchor, positive, negative = (
                int(row["anchor_idx"]), int(row["positive_idx"]), int(row["negative_idx"])
            )
            ppm = abs(float(masses[anchor]) - float(masses[negative])) / float(masses[anchor]) * 1e6
            ppm_values.append(ppm)
            ik_a, ik_p, ik_n = decode(ik[anchor])[:14], decode(ik[positive])[:14], decode(ik[negative])[:14]
            if ik_a != ik_p or ik_a == ik_n or ppm > 10.0 + 1e-7:
                protocol_errors.append((anchor, positive, negative, ppm))
            if not (decode(folds[anchor]) == decode(folds[positive]) == decode(folds[negative]) == "train"):
                protocol_errors.append((anchor, positive, negative, "fold"))
            for name in ("anchor", "positive", "negative"):
                tensor = row[name]
                if tuple(tensor.shape) != (101, 2) or not np.isfinite(tensor.numpy()).all():
                    protocol_errors.append((anchor, name, "tensor"))

    report = {
        "status": "causal_chemmask_data_smoke",
        "samples": len(rows),
        "protocol_errors": len(protocol_errors),
        "hard_negative_selected_fraction": float(np.mean([
            bool(row["hard_negative_selected"]) for row in rows
        ])),
        "identity_mask_applied_fraction": float(np.mean([
            bool(row["identity_masked"]) for row in rows
        ])),
        "masked_peak_count": {
            "min": int(min(int(row["masked_peak_count"]) for row in rows)),
            "median": float(np.median([int(row["masked_peak_count"]) for row in rows])),
            "max": int(max(int(row["masked_peak_count"]) for row in rows)),
        },
        "hard_negative_score": {
            "median": float(np.median([float(row["hard_negative_score"]) for row in rows])),
            "max": float(max(float(row["hard_negative_score"]) for row in rows)),
        },
        "negative_ppm_max": float(max(ppm_values)),
    }
    if protocol_errors:
        report["first_errors"] = protocol_errors[:10]
    output = ROOT / "data/e1/causal_chemmask_data_smoke.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if protocol_errors:
        raise RuntimeError("Causal ChemMask data smoke failed")


if __name__ == "__main__":
    main()
