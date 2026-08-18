"""Fast preflight checks for the budgeted E1 workflow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "MassSpecGym HDF5": ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    "raw SSL checkpoint": ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    "official embedding checkpoint": ROOT / "dreams/models/pretrained/embedding_model.ckpt",
    "slim official checkpoint": ROOT / "data/e1/official_embedding_slim.pt",
    "strict 10-ppm train triplet pool": ROOT / "data/e1/e1_train_triplet_pool_10ppm.npz",
    "strict 10-ppm validation triplet pool": ROOT / "data/e1/e1_val_triplet_pool_10ppm.npz",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="Treat a CPU-only environment as valid for smoke/pilot runs.",
    )
    args = parser.parse_args()
    print("E1 budget workflow preflight")
    failed = False
    files_ok = True
    for label, path in FILES.items():
        exists = path.is_file()
        failed |= not exists
        files_ok &= exists
        size = f"{path.stat().st_size / 2**20:.1f} MiB" if exists else "MISSING"
        print(f"  [{'OK' if exists else 'FAIL'}] {label}: {path} ({size})")

    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU memory free/total: {free / 2**30:.1f}/{total / 2**30:.1f} GiB")
    else:
        if args.allow_cpu:
            print("  [OK] CPU-only mode allowed for smoke/pilot; formal E1 still requires CUDA")
        else:
            print("  [FAIL] Formal E1 requires a CUDA-enabled PyTorch environment")
            failed = True

    if files_ok:
        with h5py.File(FILES["MassSpecGym HDF5"], "r") as handle:
            folds = np.asarray([
                value.decode() if isinstance(value, bytes) else str(value)
                for value in handle["fold"][:]
            ])
            iks = np.asarray([
                (value.decode() if isinstance(value, bytes) else str(value))[:14]
                for value in handle["INCHIKEY"][:]
            ])
            train_ik = set(iks[folds == "train"])
            val_ik = set(iks[folds == "val"])
            overlap = len(train_ik & val_ik)
            print(f"  Train/val IK14 overlap: {overlap}")
            failed |= overlap != 0

        for name in (
            "strict 10-ppm train triplet pool",
            "strict 10-ppm validation triplet pool",
        ):
            with np.load(FILES[name]) as pool:
                required = {"anchor_idx", "positive_ptr", "positive_idx", "negative_ptr", "negative_idx"}
                missing = required - set(pool.files)
                anchors = len(pool["anchor_idx"]) if not missing else 0
                print(f"  {name}: {anchors:,} anchors; missing keys={sorted(missing)}")
                failed |= bool(missing) or anchors == 0

        for audit_name in (
            "e1_train_triplet_pool_10ppm_audit.json",
            "e1_val_triplet_pool_10ppm_audit.json",
        ):
            audit_path = ROOT / "data/e1" / audit_name
            if not audit_path.is_file():
                print(f"  [FAIL] Missing full-edge ppm audit: {audit_path}")
                failed = True
                continue
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            positive_ok = audit["positive_edges"]["within_10ppm_fraction_of_finite"] == 1.0
            negative_ok = audit["negative_edges"]["within_10ppm_fraction_of_finite"] == 1.0
            identity_ok = (
                audit["positive_edges"]["different_ik14_edges"] == 0
                and audit["negative_edges"]["same_ik14_edges"] == 0
            )
            protocol_ok = positive_ok and negative_ok and identity_ok
            print(f"  [{'OK' if protocol_ok else 'FAIL'}] full-edge protocol: {audit_name}")
            failed |= not protocol_ok

    result = {
        "ok": not failed, "cuda": torch.cuda.is_available(),
        "cpu_mode_allowed": bool(args.allow_cpu),
    }
    out = ROOT / "data/e1/preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit("Preflight failed; fix the FAIL items before training.")
    print("Preflight passed.")


if __name__ == "__main__":
    main()
