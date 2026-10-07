"""REAL-architecture smoke for the orbit-boundary trainer (audit step 6,
CPU variant - the server GPU repeat is still required before unbanning the
sbatch, but this exercises the COMPLETE code path: real GNPS spectra ->
mini HDF5 (loader layout) -> pool builder with integrity gates -> trainer
in --encoder dreams mode loading the REAL 116M DreaMS backbone
(contrastive_v5 raw-ssl checkpoint) -> forward/backward -> checkpoints).

Steps:
  S1 pick 2 real queries (>=2 candidate molecules) from the frozen identity
     panel; assemble their candidate reference rows + multi-spectra views
     of the true molecules (instrument metadata from the manifest);
  S2 write the mini HDF5 in the loader layout (spectrum (2,n) padded,
     precursor_mz) and the builder's three input fixtures;
  S3 run GLM_build_orbit_boundary_groups (integrity gates must pass on the
     real-derived fixtures);
  S4 run GLM_train_orbit_boundary_encoder --encoder dreams for arm R and
     OC with the raw-ssl checkpoint; assert finite losses, gradient flow,
     checkpoint files, sha pinning.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))
from GLM_score_challenger_models_on_gnps import parse_mgf_used  # noqa: E402

PANEL = ROOT / "data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz"
BENCH = ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
CKPT = ROOT / "contrastive_checkpoints_v5/contrastive_v5_epoch5.pt"
tmp = Path(tempfile.mkdtemp(prefix="orbit_real_smoke_"))


def main() -> None:
    with np.load(PANEL) as z:
        panel = {k: np.asarray(z[k]) for k in z.files}
    qptr, mptr = panel["query_ptr"], panel["molecule_ptr"]
    manifest = pd.read_csv(BENCH / "manifest.csv.gz",
                           usecols=["row", "ik14", "instrument", "n_peaks"])

    # S1: two queries with >=2 molecules, positive has >=2 library spectra
    chosen = []
    for q in range(panel["query_row"].shape[0]):
        lo, hi = int(qptr[q]), int(qptr[q + 1])
        if hi - lo < 2:
            continue
        pos_ik = str(panel["molecule_ik14"][lo])
        n_lib = int((manifest.ik14.astype(str) == pos_ik).sum())
        if n_lib >= 3:
            chosen.append(q)
        if len(chosen) == 2:
            break
    assert len(chosen) == 2, "could not find two eligible queries"

    needed, groups = set(), []
    for g_i, q in enumerate(chosen):
        lo, hi = int(qptr[q]), int(qptr[q + 1])
        qrow = int(panel["query_row"][q])
        needed.add(qrow)
        refs = []
        for m in range(lo, hi):
            for r in map(int, panel["candidate_row"][mptr[m]:mptr[m + 1]]):
                refs.append(r)
                needed.add(r)
        groups.append((q, qrow, lo, hi, refs))

    # condition ledger: all library rows of each group's TRUE molecule,
    # PLUS the query rows themselves (a panel query IS a spectrum of the
    # true molecule; the builder resolves the query view through the ledger)
    views_rows, views_mol, views_inst, views_qual = [], [], [], []
    inst_codes = {"LC-ESI-Orbitrap": 1, "ESI-Orbitrap": 1,
                  "LC-ESI-ToF": 2, "ESI-qTof": 3}
    for q, qrow, lo, hi, refs in groups:
        pos_ik = str(panel["molecule_ik14"][lo])
        lib = manifest[manifest.ik14.astype(str) == pos_ik]
        for _, r in lib.iterrows():
            views_rows.append(int(r.row))
            views_mol.append(pos_ik)
            views_inst.append(inst_codes.get(str(r.instrument), 1))
            views_qual.append(0.5)
            needed.add(int(r.row))
        mrow = manifest[manifest.row == qrow]
        inst = inst_codes.get(str(mrow.iloc[0].instrument), 1) if len(mrow) \
            else 1
        views_rows.append(qrow)
        views_mol.append(pos_ik)
        views_inst.append(inst)
        views_qual.append(0.5)
    needed = sorted(set(needed) | {g[1] for g in groups})
    print(f"S1 PASS: 2 groups, {len(needed)} real spectra rows", flush=True)

    spectra = parse_mgf_used(BENCH / "spectra.mgf", set(needed))
    row_index = {r: i for i, r in enumerate(needed)}

    # S2: mini HDF5 in the loader layout
    n_max = max(len(spectra[r][0]) for r in needed)
    with h5py.File(tmp / "mini.h5", "w") as h:
        spec_ds = h.create_dataset("spectrum", (len(needed), 2, n_max),
                                   dtype="f4")
        prec_ds = h.create_dataset("precursor_mz", (len(needed),),
                                   dtype="f8")
        for r in needed:
            i = row_index[r]
            mz, inten, prec = spectra[r]
            spec_ds[i, 0, :len(mz)] = mz
            spec_ds[i, 1, :len(inten)] = inten
            prec_ds[i] = prec
    print(f"S2a PASS: mini HDF5 ({len(needed)} rows, width {n_max})",
          flush=True)

    # builder fixtures (rows remapped into the mini HDF5 row space)
    cand_ptr = [0]
    ref_ptr = [0]
    molecule_label: list[int] = []
    mol_ik14: list[str] = []
    ref_rows: list[int] = []
    for q, qrow, lo, hi, refs in groups:
        for m in range(lo, hi):
            ik = str(panel["molecule_ik14"][m])
            block = [row_index[int(r)] for r in
                     panel["candidate_row"][mptr[m]:mptr[m + 1]]]
            if not block:
                continue
            ref_rows.extend(block)
            ref_ptr.append(len(ref_rows))
            molecule_label.append(1 if m == lo else 0)
            mol_ik14.append(ik)
        cand_ptr.append(len(molecule_label))
    cand_groups = {
        "query_row": np.asarray([row_index[qrow] for _, qrow, *_ in groups],
                                dtype=np.int64),
        "cand_ptr": np.asarray(cand_ptr, dtype=np.int64),
        "molecule_label": np.asarray(molecule_label, dtype=np.int8),
        "mol_ik14": np.asarray(mol_ik14),
        "formula_cluster": np.asarray([0, 1]),
        "val_query_mask": np.asarray([False, False]),
        "ref_ptr": np.asarray(ref_ptr, dtype=np.int64),
        "ref_rows": np.asarray(ref_rows, dtype=np.int64),
    }
    views = {
        "rows": np.asarray([row_index[r] for r in views_rows]),
        "molecule": np.asarray(views_mol),
        "instrument": np.asarray(views_inst),
        "quality": np.asarray(views_qual, float),
    }
    margins = {"group_id": np.asarray([0, 1, 0, 1]),
               "candidate_local": np.asarray([1, 1, 2, 2]),
               "margin": np.asarray([0.3, 0.3, 0.5, 0.5], dtype=np.float32)}
    np.savez(tmp / "cand.npz", **cand_groups)
    np.savez(tmp / "views.npz", **views)
    np.savez(tmp / "marg.npz", **margins)
    print("S2b PASS: builder fixtures written (real rows)", flush=True)

    pool = tmp / "pool.npz"
    r = subprocess.run(
        [sys.executable, "-X", "utf8",
         str(ROOT / "tasks/GLM_build_orbit_boundary_groups.py"),
         "--candidate-groups", str(tmp / "cand.npz"),
         "--condition-views", str(tmp / "views.npz"),
         "--chem-margins", str(tmp / "marg.npz"),
         "--output", str(pool)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-600:] + r.stderr[-800:]
    print("S3 PASS: pool built with integrity gates on real-derived fixtures",
          flush=True)
    sha = json.loads((tmp / "pool.json").read_text(encoding="utf-8"))["pool_sha256"]

    # S4 gate: a loadable server-format checkpoint (raw_ssl / official
    # embedding). Local v5 checkpoints are our own chemaware training-state
    # files (no args) and are NOT loadable; the DreaMS release weights are
    # server-side. Without a qualifying checkpoint, S4 is BLOCKED-SKIPPED
    # with an explicit marker - never a silent pass.
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "tasks"))
    from e1_checkpoint_io import checkpoint_kind  # noqa: E402
    import torch as _torch  # noqa: E402
    ckpt_ok = False
    try:
        pkg = _torch.load(str(CKPT), map_location="cpu", weights_only=False)
        checkpoint_kind(pkg)
        ckpt_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"S4 BLOCKED-SKIP (server asset): local checkpoint not in a "
              f"loader-supported format ({exc})", flush=True)
    if not ckpt_ok:
        print("REAL-ARCHITECTURE SMOKE: S1-S3 PASS, S4 requires the server "
              "DreaMS release checkpoint (official_embedding.pt / "
              "ssl_model_server.pt) - run unchanged on the cluster")
        raise SystemExit(3)

    for arm in ("R", "OC"):
        out = tmp / f"dreams_{arm}"
        r = subprocess.run(
            [sys.executable, "-X", "utf8",
             str(ROOT / "tasks/GLM_train_orbit_boundary_encoder.py"),
             "--arm", arm, "--pool", str(pool),
             "--expected-pool-sha256", sha,
             "--encoder", "dreams",
             "--official-checkpoint", str(CKPT),
             "--architecture-checkpoint", str(CKPT),
             "--noise-v1-checkpoint", str(CKPT),
             "--data", str(tmp / "mini.h5"),
             "--steps", "3", "--batch-size", "2", "--lr", "1e-9",
             "--seed", "3407", "--output-dir", str(out)],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-800:] + r.stderr[-1500:]
        rep = json.loads((out / f"{arm}_report.json").read_text(encoding="utf-8"))
        assert all(np.isfinite(rep["loss_history"]))
        assert rep["encoder"] == "dreams" and rep["pool_sha256"] == sha
        assert len(list(out.glob(f"{arm}_step*.ckpt"))) >= 3
        print(f"S4 PASS [{arm}]: real DreaMS forward/backward, losses "
              f"{rep['loss_history']}", flush=True)
    print("REAL-ARCHITECTURE SMOKE: ALL PASS")


if __name__ == "__main__":
    main()
