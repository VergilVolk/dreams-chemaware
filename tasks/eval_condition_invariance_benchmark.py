"""Locked condition-invariance benchmark for the cross-condition FN error.

This is the "ruler" for the FN task: does a checkpoint keep the *same molecule*
close in embedding space when the acquisition condition (instrument / collision
energy) changes, while still keeping *different molecules* apart?

The metric mirrors strict 10-ppm retrieval, but with the positive made hard:

    cross_cosine   = cos(same-molecule, cross-condition pair)      [FN signal]
    negative_cosine = cos(different-molecule, condition-matched)   [FP guard]
    margin          = cross_cosine − negative_cosine               [retrieval sep]

Plus a secondary floor reference: same-molecule *same-condition* cosine (true
technical replicates, rare in real data) giving the ceiling invariance can
reach, and gap = floor − cross (the condition-induced degradation).

Measured on the FINAL (post-head, L2-normalized) embedding — the vector strict
10-ppm retrieval actually compares, and the object head fine-tuning changes.
Uncertainty is molecule-clustered bootstrap (one pair per molecule).

The cohort is locked: real spectra only (SIMULATION_CHALLENGE == "False"), val
fold, same adduct, deterministic sampling, and its manifest is written once
with a sha256 so the ruler cannot silently drift.

Run `--build-only` to (re)lock the cohort without embedding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from e1_checkpoint_io import (  # noqa: E402
    checkpoint_kind,
    official_backbone_state,
    official_head_state,
    torch_load_compat,
)
from pilot_paired_layer_cka import SpectrumRows  # noqa: E402  (preprocessing)
from train_e1_identity import load_base_model  # noqa: E402


DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_RAW = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OFFICIAL = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_OUTPUT = ROOT / "data/validation/condition_invariance_benchmark"
LOCK_SEED = 20260816


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--raw-checkpoint", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument(
        "--head-checkpoint",
        type=Path,
        default=None,
        help="Optional fine-tuned head checkpoint (causal/counterfactual). "
        "If omitted, the official embedding head is measured.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", default="val")
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--max-members", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Write the locked cohort manifest and exit without embedding.",
    )
    return parser.parse_args()


def representative_members(
    members: list[int],
    instrument: np.ndarray,
    collision_energy: np.ndarray,
    max_members: int,
) -> list[int]:
    """Bound replicate-group size while keeping each instrument's CE extremes."""
    if len(members) <= max_members:
        return sorted(members)
    selected: set[int] = set()
    members_arr = np.asarray(members, dtype=np.int64)
    for inst in np.unique(instrument[members_arr]):
        subset = members_arr[instrument[members_arr] == inst]
        finite = subset[np.isfinite(collision_energy[subset])]
        if len(finite):
            selected.add(int(finite[np.argmin(collision_energy[finite])]))
            selected.add(int(finite[np.argmax(collision_energy[finite])]))
        selected.add(int(subset[0]))
    remaining = [m for m in members if m not in selected]
    step = max(1, len(remaining) // max(0, max_members - len(selected)))
    selected.update(remaining[::step][: max(0, max_members - len(selected))])
    return sorted(selected)


def build_locked_cohort(data_path: Path, fold: str, max_members: int) -> dict:
    """Build same-adduct same-molecule pairs + condition-matched negatives.

    Returns:
      cross_pairs — one per molecule (instrument differs or CE delta >= 10),
          each carrying a condition-matched different-molecule negative row.
      same_pairs — one per molecule (same instrument, CE delta < 10), the rare
          technical-replicate floor reference.
    """
    rng = np.random.RandomState(LOCK_SEED)
    cross_pairs: list[dict] = []
    same_pairs: list[dict] = []

    with h5py.File(data_path, "r") as handle:
        folds = handle["fold"].asstr()[:]
        sim = handle["SIMULATION_CHALLENGE"].asstr()[:]
        ik = handle["INCHIKEY"].asstr()[:]
        instrument = handle["INSTRUMENT_TYPE"].asstr()[:]
        adduct = handle["adduct"].asstr()[:]
        collision_energy = np.asarray(handle["COLLISION_ENERGY"][:])
        precursor_mz = np.asarray(handle["precursor_mz"][:])

        valid = (
            (folds == fold)
            & (sim == "False")
            & np.isfinite(precursor_mz)
            & (precursor_mz > 0)
            & (precursor_mz <= 1000)
        )
        rows = np.flatnonzero(valid)

        # Groups of same-molecule, same-adduct spectra.
        groups: dict[tuple, list[int]] = defaultdict(list)
        for row in rows:
            groups[(ik[row][:14], adduct[row])].append(int(row))

        # Negative-matching index: (instrument, adduct) -> rows, then adduct -> rows.
        inst_add: dict[tuple, list[int]] = defaultdict(list)
        add_only: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            inst_add[(instrument[row], adduct[row])].append(int(row))
            add_only[adduct[row]].append(int(row))

        def match_negative(anchor_row: int, own_ik14: str) -> int:
            """Nearest-m/z different molecule, matching instrument+adduct first."""
            for pool in (
                inst_add.get((instrument[anchor_row], adduct[anchor_row]), []),
                add_only.get(adduct[anchor_row], []),
            ):
                candidates = [
                    r for r in pool
                    if ik[r][:14] != own_ik14 and r != anchor_row
                ]
                if candidates:
                    return min(
                        candidates,
                        key=lambda r: abs(precursor_mz[r] - precursor_mz[anchor_row]),
                    )
            raise RuntimeError(f"No negative candidate for row {anchor_row}")

        for (ik14, _adduct), members in groups.items():
            if len(members) < 2:
                continue
            members = representative_members(members, instrument, collision_energy, max_members)
            best_cross: tuple | None = None  # (score, i, j)
            best_same: tuple | None = None  # (ce_delta, i, j)
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = members[a], members[b]
                    inst_diff = instrument[i] != instrument[j]
                    cei, cej = collision_energy[i], collision_energy[j]
                    both_ce = np.isfinite(cei) and np.isfinite(cej)
                    ce_delta = abs(cei - cej) if both_ce else None
                    if inst_diff or (both_ce and ce_delta >= 10):
                        score = 5.0 * inst_diff
                        if both_ce:
                            score += 3.0 + min(ce_delta, 80.0) / 80.0
                        score += rng.uniform(0, 1e-6)
                        if best_cross is None or score > best_cross[0]:
                            best_cross = (score, i, j)
                    if (not inst_diff) and both_ce and ce_delta < 10:
                        if best_same is None or ce_delta < best_same[0]:
                            best_same = (ce_delta, i, j)

            def to_pair(ik14: str, i: int, j: int, kind: str) -> dict:
                return {
                    "ik14": ik14,
                    "kind": kind,
                    "rows": [i, j],
                    "instrument": [instrument[i], instrument[j]],
                    "adduct": [adduct[i], adduct[j]],
                    "collision_energy": [
                        None if not np.isfinite(collision_energy[i]) else float(collision_energy[i]),
                        None if not np.isfinite(collision_energy[j]) else float(collision_energy[j]),
                    ],
                    "precursor_mz": [float(precursor_mz[i]), float(precursor_mz[j])],
                }

            if best_cross is not None:
                pair = to_pair(ik14, best_cross[1], best_cross[2], "cross")
                pair["negative_row"] = match_negative(best_cross[1], ik14)
                cross_pairs.append(pair)
            if best_same is not None:
                same_pairs.append(to_pair(ik14, best_same[1], best_same[2], "same"))

    manifest = {
        "status": "condition_invariance_benchmark_cohort",
        "lock_seed": LOCK_SEED,
        "fold": fold,
        "simulation_challenge": "False only",
        "adduct": "same-adduct only",
        "cross_pairs": cross_pairs,
        "same_pairs": same_pairs,
        "audit": {
            "n_cross_pairs": len(cross_pairs),
            "n_same_pairs": len(same_pairs),
        },
    }
    return manifest


def row_cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    num = np.sum(x * y, axis=-1)
    den = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return num / np.clip(den, 1e-12, None)


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def molecule_bootstrap(
    values: dict[str, np.ndarray],
    rng: np.random.RandomState,
    n_bootstrap: int,
) -> dict:
    """Bootstrap several per-molecule quantities with a shared resampling."""
    n = len(next(iter(values.values())))
    samples = rng.randint(0, n, size=(n_bootstrap, n))
    out = {}
    for key, arr in values.items():
        boot = arr[samples].mean(axis=1)
        out[key] = {
            "mean": float(arr.mean()),
            "ci95": percentile_ci(boot),
        }
    return out


def embed_rows(
    model: torch.nn.Module,
    rows: np.ndarray,
    data_path: Path,
    n_highest_peaks: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        SpectrumRows(data_path, rows, n_highest_peaks),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model_dtype = next(model.backbone.parameters()).dtype
    embeddings = []
    with torch.inference_mode():
        for spectra in loader:
            spectra = spectra.to(device=device, dtype=model_dtype)
            embeddings.append(model(spectra).float().cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "pairs.json"

    manifest = build_locked_cohort(args.data, args.fold, args.max_members)
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8")
    if manifest_path.exists():
        existing = manifest_path.read_bytes()
        if hashlib.sha256(existing).digest() != hashlib.sha256(manifest_bytes).digest():
            raise RuntimeError(
                f"Locked cohort changed vs {manifest_path}. If intentional, delete "
                "the file or move it, then re-run to relock."
            )
    else:
        manifest_path.write_bytes(manifest_bytes)

    print(
        f"Locked cohort: {manifest['audit']['n_cross_pairs']} cross, "
        f"{manifest['audit']['n_same_pairs']} same-condition floor pairs "
        f"(fold={args.fold}, real-only, same-adduct)",
        flush=True,
    )

    if args.build_only:
        print(f"Manifest written: {manifest_path}", flush=True)
        return

    all_rows = []
    for pair in manifest["cross_pairs"]:
        all_rows.extend(pair["rows"])
        all_rows.append(pair["negative_row"])
    for pair in manifest["same_pairs"]:
        all_rows.extend(pair["rows"])
    unique_rows, _ = np.unique(np.asarray(all_rows, dtype=np.int64), return_inverse=True)
    idx_map = {int(r): k for k, r in enumerate(unique_rows)}

    print("Loading base model (official backbone + head)...", flush=True)
    model, init_kind = load_base_model(
        args.official_checkpoint, args.raw_checkpoint, device, args.n_highest_peaks
    )
    tag = f"official_{init_kind}"
    if args.head_checkpoint is not None:
        package = torch_load_compat(args.head_checkpoint, map_location="cpu")
        kind = checkpoint_kind(package)
        model.head.load_state_dict(official_head_state(package), strict=True)
        model.backbone.load_state_dict(official_backbone_state(package), strict=True)
        tag = f"{kind}_epoch{int(package.get('epoch', -1))}"
    model.eval()

    print(f"Embedding {len(unique_rows)} spectra ({tag})...", flush=True)
    embeddings = embed_rows(
        model, unique_rows, args.data, args.n_highest_peaks, args.batch_size, device
    )

    cross_i = np.asarray([idx_map[p["rows"][0]] for p in manifest["cross_pairs"]])
    cross_j = np.asarray([idx_map[p["rows"][1]] for p in manifest["cross_pairs"]])
    neg_k = np.asarray([idx_map[p["negative_row"]] for p in manifest["cross_pairs"]])
    same_i = np.asarray([idx_map[p["rows"][0]] for p in manifest["same_pairs"]])
    same_j = np.asarray([idx_map[p["rows"][1]] for p in manifest["same_pairs"]])

    cross_cos = row_cosine(embeddings[cross_i], embeddings[cross_j])
    negative_cos = row_cosine(embeddings[cross_i], embeddings[neg_k])
    margin = cross_cos - negative_cos
    same_cos = row_cosine(embeddings[same_i], embeddings[same_j]) if len(same_i) else None

    # Stratify cross pairs by the condition that actually differs, so the CE
    # gap (the largest FN source) is tracked separately instead of being
    # averaged away by the more common instrument-difference pairs.
    cond_type = []
    for p in manifest["cross_pairs"]:
        inst_diff = p["instrument"][0] != p["instrument"][1]
        ce = p["collision_energy"]
        ce_diff = ce[0] is not None and ce[1] is not None and abs(ce[0] - ce[1]) >= 10
        cond_type.append(
            "both" if inst_diff and ce_diff else
            "instrument" if inst_diff else "ce"
        )
    cond_type = np.asarray(cond_type)

    rng = np.random.RandomState(LOCK_SEED + 7)
    values = {
        "cross_cosine": cross_cos,
        "negative_cosine": negative_cos,
        "margin": margin,
    }
    boot = molecule_bootstrap(values, rng, args.n_bootstrap)

    stratified = {}
    for name in ("instrument", "ce", "both"):
        mask = cond_type == name
        if not mask.sum():
            continue
        stratified[name] = {
            "n_pairs": int(mask.sum()),
            **molecule_bootstrap(
                {
                    "cross_cosine": cross_cos[mask],
                    "negative_cosine": negative_cos[mask],
                    "margin": margin[mask],
                },
                rng, args.n_bootstrap,
            ),
        }

    result = {
        "status": "condition_invariance_benchmark_result",
        "checkpoint": str((args.head_checkpoint or args.official_checkpoint).resolve()),
        "checkpoint_tag": tag,
        "initialization_kind": init_kind,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()[:16],
        "fold": args.fold,
        "n_cross_pairs": int(len(cross_cos)),
        "n_negative_matched": int(len(negative_cos)),
        "n_same_condition_floor_pairs": int(len(same_i)),
        "metrics": boot,
        "stratified_by_condition": stratified,
        "same_condition_floor": (
            None if same_cos is None else molecule_bootstrap(
                {"same_condition_cosine": same_cos}, rng, args.n_bootstrap
            )
        ),
        "gap_vs_floor": (
            None if same_cos is None else float(same_cos.mean() - cross_cos.mean())
        ),
        "interpretation": (
            "FN fix = raise cross_cosine (same molecule across conditions) toward the "
            "same-condition floor, while keeping negative_cosine (different molecule, "
            "condition-matched) from rising — i.e. widen margin. gap_vs_floor is the "
            "condition-induced degradation, secondary because floor pairs are rare."
        ),
    }

    result_path = args.output_dir / f"result_{tag}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {result_path}", flush=True)


if __name__ == "__main__":
    main()
