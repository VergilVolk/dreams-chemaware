"""Small, checkpoint-paired audit of DreaMS layer representations.

This script is deliberately a *pipeline validation*, not a factor-discovery
experiment.  The same spectra are passed through the raw SSL backbone and the
official contrastively fine-tuned backbone.  We capture the precursor token
after each complete Transformer block (attention, feed-forward, and residual
connections), then compute a layer-by-layer linear CKA matrix.

The models are loaded and evaluated sequentially so the CPU smoke test does
not need to keep two 116M-parameter backbones in memory at once.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import MethodType

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from e1_checkpoint_io import (  # noqa: E402
    checkpoint_kind,
    official_backbone_state,
    torch_load_compat,
)


DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_RAW = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_OFFICIAL = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_OUTPUT = ROOT / "data/validation/paired_layer_cka_pilot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired raw-vs-official DreaMS layer extraction and CKA pilot"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--raw-checkpoint", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", default="val")
    parser.add_argument("--n-spectra", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--verify-determinism",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Repeat the first batch; useful on GPU but expensive on this CPU host",
    )
    return parser.parse_args()


def decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def select_stratified_indices(
    data_path: Path, fold: str, n_spectra: int, seed: int
) -> tuple[np.ndarray, list[dict]]:
    """Choose a small deterministic sample spread over precursor-mass bins.

    This is enough for a pipeline smoke test.  It is intentionally not the
    formal 30k--50k balanced discovery cohort.
    """
    rng = np.random.RandomState(seed)
    with h5py.File(data_path, "r") as handle:
        folds = np.asarray([decode_text(v) for v in handle["fold"][:]])
        precursor = np.asarray(handle["precursor_mz"][:], dtype=np.float64)
        valid = np.isfinite(precursor) & (precursor > 0) & (precursor <= 1000)
        if fold.lower() != "all":
            valid &= folds == fold
        candidates = np.flatnonzero(valid)
        if len(candidates) < n_spectra:
            raise ValueError(
                f"Only {len(candidates):,} valid spectra in fold={fold!r}; "
                f"requested {n_spectra:,}. Available folds: "
                f"{sorted(set(folds.tolist()))}"
            )

        # Equal-frequency precursor-mass strata prevent the tiny pilot from
        # accidentally occupying only one narrow mass range.
        candidate_mz = precursor[candidates]
        quantiles = np.quantile(candidate_mz, np.linspace(0, 1, 9))
        bins = np.unique(quantiles)
        if len(bins) < 3:
            chosen = rng.choice(candidates, size=n_spectra, replace=False)
        else:
            bin_id = np.clip(np.digitize(candidate_mz, bins[1:-1]), 0, len(bins) - 2)
            groups = [candidates[bin_id == b] for b in range(len(bins) - 1)]
            for group in groups:
                rng.shuffle(group)
            cursors = np.zeros(len(groups), dtype=int)
            selected: list[int] = []
            while len(selected) < n_spectra:
                made_progress = False
                for b, group in enumerate(groups):
                    if cursors[b] < len(group) and len(selected) < n_spectra:
                        selected.append(int(group[cursors[b]]))
                        cursors[b] += 1
                        made_progress = True
                if not made_progress:
                    break
            chosen = np.asarray(selected, dtype=np.int64)

        chosen = np.sort(chosen)
        metadata = []
        for row in chosen:
            metadata.append({
                "row": int(row),
                "fold": decode_text(handle["fold"][row]),
                "inchikey": decode_text(handle["INCHIKEY"][row]),
                "precursor_mz": float(handle["precursor_mz"][row]),
                "instrument": decode_text(handle["INSTRUMENT_TYPE"][row]),
                "collision_energy": float(handle["COLLISION_ENERGY"][row]),
                "adduct": decode_text(handle["adduct"][row]),
            })
    return chosen, metadata


def preprocess_spectrum(
    raw_2_n: np.ndarray, precursor_mz: float, n_highest: int
) -> torch.Tensor:
    """Match the established MassSpecGym preprocessing used by E0/E1."""
    raw = np.asarray(raw_2_n)
    highest = np.argsort(raw[1], kind="stable")[-n_highest:]
    highest = np.sort(highest)
    peaks = raw[:, highest].T.astype(np.float32, copy=True)
    if len(peaks) < n_highest:
        peaks = np.pad(peaks, ((0, n_highest - len(peaks)), (0, 0)))
    maximum = float(peaks[:, 1].max())
    if maximum > 0:
        peaks[:, 1] /= maximum
    precursor = np.asarray([[precursor_mz, 1.1]], dtype=np.float32)
    return torch.from_numpy(np.vstack((precursor, peaks)))


class SpectrumRows(Dataset):
    def __init__(self, path: Path, rows: np.ndarray, n_highest_peaks: int):
        self.path = str(path)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.n_highest_peaks = n_highest_peaks
        self._handle = None

    def __len__(self) -> int:
        return len(self.rows)

    def _h5(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, item: int) -> torch.Tensor:
        handle = self._h5()
        row = int(self.rows[item])
        return preprocess_spectrum(
            handle["spectrum"][row],
            float(handle["precursor_mz"][row]),
            self.n_highest_peaks,
        )

    def __del__(self):
        if self._handle is not None:
            self._handle.close()


class PositionFeedForward(nn.Module):
    """Inference-only copy of the DreaMS positional feed-forward module.

    Keeping the same ``ff.*`` module names makes its checkpoint keys identical,
    while avoiding imports of plotting and training dependencies.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        bias: bool,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        hidden = [hidden_dim] * depth
        for layer in range(depth):
            d_in = hidden[layer - 1] if layer else in_dim
            d_out = hidden[layer] if layer != depth - 1 else out_dim
            layers.append(nn.Linear(d_in, d_out, bias=bias))
            if layer != depth - 1:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
        self.ff = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class LightweightDreaMS(nn.Module):
    """Exact inference path of the checkpoint without Lightning utilities."""
    def __init__(self, args: Namespace):
        super().__init__()
        if args.vanilla_transformer:
            raise NotImplementedError("The audited checkpoints use the custom encoder")
        if not args.d_fourier or args.d_mz_token:
            raise NotImplementedError("This pilot is validated for the Fourier DreaMS model")
        if args.charge_feature:
            raise NotImplementedError("The audited checkpoints do not use charge features")

        # These two local modules depend only on torch. Importing the full
        # Lightning DreaMS class brings in several unrelated training stacks.
        from dreams.models.layers.fourier_features import FourierFeatures
        from dreams.models.dreams.layers import TransformerEncoder

        self.n_layers = int(args.n_layers)
        self.d_model = int(args.d_fourier + args.d_peak + args.d_mz_token)
        self.max_mz = float(args.dformat.max_mz)
        self.graphormer_mz_diffs = bool(args.graphormer_mz_diffs)
        args.d_model = self.d_model
        args.d_graphormer_params = 0

        self.fourier_enc = FourierFeatures(
            strategy=args.fourier_strategy,
            num_freqs=args.fourier_num_freqs,
            x_min=(
                args.dformat.max_tbxic_stdev
                if not args.fourier_min_freq else args.fourier_min_freq
            ),
            x_max=args.dformat.max_mz,
            trainable=args.fourier_trainable,
        )
        self.ff_fourier = PositionFeedForward(
            in_dim=self.fourier_enc.num_features(),
            out_dim=args.d_fourier,
            hidden_dim=args.ff_fourier_d,
            depth=args.ff_fourier_depth,
            dropout=args.dropout,
            bias=not args.no_ffs_bias,
        )
        self.ff_peak = PositionFeedForward(
            in_dim=2,
            out_dim=args.d_peak,
            hidden_dim=args.d_peak,
            depth=args.ff_peak_depth,
            dropout=args.dropout,
            bias=not args.no_ffs_bias,
        )
        self.transformer_encoder = TransformerEncoder(args)

    def forward(self, spec: torch.Tensor, charge=None) -> torch.Tensor:
        padding_mask = spec[:, :, 0] == 0
        normalizer = torch.tensor(
            [self.max_mz, 1.0], device=spec.device, dtype=spec.dtype
        )
        peak_embeddings = self.ff_peak(spec / normalizer)
        fourier_features = self.ff_fourier(self.fourier_enc(spec[..., [0]]))
        tokens = torch.cat([peak_embeddings, fourier_features], dim=-1)
        graphormer_dists = None
        if self.graphormer_mz_diffs:
            # DreaMS still passes Fourier pair differences when the learned
            # Graphormer projection is disabled; attention sums the last axis.
            graphormer_dists = (
                fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)
            )
        return self.transformer_encoder(tokens, padding_mask, graphormer_dists)


def reconstruct_backbone(
    architecture_package: dict,
    state_dict: dict[str, torch.Tensor],
    n_highest_peaks: int,
    device: torch.device,
):
    started = time.time()
    print("    reconstructing lightweight inference backbone", flush=True)

    model_args = Namespace(**architecture_package["args"])
    model_args.dformat = Namespace(
        max_mz=float(architecture_package["args"]["max_mz"]),
        max_tbxic_stdev=float(
            architecture_package["args"]["max_tbxic_stdev"]
        ),
    )
    print(f"    constructing {model_args.n_layers}-layer backbone", flush=True)
    backbone = LightweightDreaMS(model_args)
    print(f"    backbone constructed in {time.time() - started:.1f}s", flush=True)
    load_started = time.time()
    # ff_out and ro_out are training-only prediction heads and are never used
    # in the embedding forward pass. All remaining forward-path keys are
    # required to match exactly.
    ignored_prefixes = ("ff_out.", "ro_out.")
    forward_state = {
        key: value for key, value in state_dict.items()
        if not key.startswith(ignored_prefixes)
    }
    incompatible = backbone.load_state_dict(forward_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    print(f"    state loaded in {time.time() - load_started:.1f}s", flush=True)
    move_started = time.time()
    backbone.eval().to(device)
    print(f"    model moved to {device} in {time.time() - move_started:.1f}s", flush=True)
    return backbone


def extract_complete_block_outputs(
    backbone: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    verify_determinism: bool,
) -> tuple[np.ndarray, dict]:
    """Capture precursor activations after each full Transformer block.

    A hook on ``ffs[i]`` would capture only the feed-forward branch *before*
    its residual addition.  Wrapping ``_layer_forward`` records the actual
    residual-stream output of the complete block.
    """
    encoder = backbone.transformer_encoder
    if not hasattr(encoder, "_layer_forward"):
        raise TypeError("This pilot currently requires the custom DreaMS encoder")
    original = encoder._layer_forward
    current: dict[int, torch.Tensor] = {}

    def wrapped(this, layer_index, *args, **kwargs):
        output = original(layer_index, *args, **kwargs)
        current[int(layer_index)] = output[:, 0, :].detach().float().cpu()
        return output

    encoder._layer_forward = MethodType(wrapped, encoder)
    batches = []
    determinism_max_abs = None
    started = time.time()
    model_dtype = next(backbone.parameters()).dtype
    try:
        with torch.inference_mode():
            for batch_index, spectra in enumerate(loader):
                spectra = spectra.to(device=device, dtype=model_dtype)
                current.clear()
                final = backbone(spectra, None)[:, 0, :].detach().float().cpu()
                if sorted(current) != list(range(backbone.n_layers)):
                    raise RuntimeError(
                        f"Captured layers {sorted(current)}, expected "
                        f"0..{backbone.n_layers - 1}"
                    )
                layer_batch = torch.stack(
                    [current[i] for i in range(backbone.n_layers)], dim=1
                )
                # The encoder applies a final normalization after block 7.
                # Store that exact returned representation for the final layer.
                layer_batch[:, -1, :] = final

                if batch_index == 0 and verify_determinism:
                    current.clear()
                    repeat = backbone(spectra, None)[:, 0, :].detach().float().cpu()
                    determinism_max_abs = float((repeat - final).abs().max())
                batches.append(layer_batch.numpy())
    finally:
        encoder._layer_forward = original

    activations = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    diagnostics = {
        "shape": list(activations.shape),
        "finite_fraction": float(np.isfinite(activations).mean()),
        "determinism_max_abs": determinism_max_abs,
        "seconds": time.time() - started,
        "definition": (
            "precursor token after each complete Transformer block; final layer "
            "uses the backbone's final normalized output"
        ),
    }
    return activations, diagnostics


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Centered linear CKA in feature space (Kornblith et al., 2019)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    if len(x) < min(x.shape[1], y.shape[1]):
        # The pilot has far fewer spectra than features. The sample-space
        # expression is exactly equivalent and avoids 1024x1024 products.
        gram_x = x @ x.T
        gram_y = y @ y.T
        numerator = float((gram_x * gram_y).sum())
        denominator = float(
            np.sqrt(np.square(gram_x).sum() * np.square(gram_y).sum())
        )
        return numerator / denominator if denominator > 0 else float("nan")
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = float(np.square(cross).sum())
    denominator = float(
        np.sqrt(np.square(xx).sum() * np.square(yy).sum())
    )
    return numerator / denominator if denominator > 0 else float("nan")


def cka_matrix(raw: np.ndarray, official: np.ndarray) -> np.ndarray:
    n_raw = raw.shape[1]
    n_official = official.shape[1]
    matrix = np.empty((n_raw, n_official), dtype=np.float64)
    for i in range(n_raw):
        for j in range(n_official):
            matrix[i, j] = linear_cka(raw[:, i], official[:, j])
    return matrix


def paired_cosine(raw: np.ndarray, official: np.ndarray) -> list[float]:
    result = []
    for layer in range(raw.shape[1]):
        x = raw[:, layer].astype(np.float64)
        y = official[:, layer].astype(np.float64)
        denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        values = np.divide(
            np.sum(x * y, axis=1), denominator,
            out=np.full(len(x), np.nan), where=denominator > 0,
        )
        result.append(float(np.nanmean(values)))
    return result


def main() -> None:
    args = parse_args()
    if args.n_spectra < 4:
        raise ValueError("--n-spectra must be at least 4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but this PyTorch environment has no CUDA")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Selecting the paired spectrum cohort", flush=True)
    rows, metadata = select_stratified_indices(
        args.data, args.fold, args.n_spectra, args.seed
    )
    loader = DataLoader(
        SpectrumRows(args.data, rows, args.n_highest_peaks),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(
        f"  {len(rows)} spectra; precursor m/z "
        f"{min(m['precursor_mz'] for m in metadata):.2f}--"
        f"{max(m['precursor_mz'] for m in metadata):.2f}",
        flush=True,
    )

    print("[2/5] Loading architecture and raw SSL backbone", flush=True)
    raw_package = torch_load_compat(args.raw_checkpoint, map_location="cpu")
    if checkpoint_kind(raw_package) != "raw_ssl":
        raise ValueError("--raw-checkpoint must be ssl_model_server.pt format")
    raw_model = reconstruct_backbone(
        raw_package,
        raw_package["state_dict"],
        args.n_highest_peaks,
        device,
    )
    raw_activations, raw_diagnostics = extract_complete_block_outputs(
        raw_model, loader, device, args.verify_determinism
    )
    del raw_model
    gc.collect()
    print(f"  raw shape={raw_activations.shape}; {raw_diagnostics['seconds']:.1f}s", flush=True)

    print("[3/5] Loading official fine-tuned backbone", flush=True)
    official_package = torch_load_compat(args.official_checkpoint, map_location="cpu")
    if checkpoint_kind(official_package) not in (
        "official_embedding", "official_embedding_slim"
    ):
        raise ValueError("--official-checkpoint must be an official embedding checkpoint")
    official_model = reconstruct_backbone(
        raw_package,
        official_backbone_state(official_package),
        args.n_highest_peaks,
        device,
    )
    official_activations, official_diagnostics = extract_complete_block_outputs(
        official_model, loader, device, args.verify_determinism
    )
    del official_model, official_package, raw_package
    gc.collect()
    print(
        f"  official shape={official_activations.shape}; "
        f"{official_diagnostics['seconds']:.1f}s",
        flush=True,
    )

    print("[4/5] Computing paired layer diagnostics", flush=True)
    if raw_activations.shape != official_activations.shape:
        raise RuntimeError(
            f"Activation shape mismatch: {raw_activations.shape} vs "
            f"{official_activations.shape}"
        )
    matrix = cka_matrix(raw_activations, official_activations)
    diagonal = np.diag(matrix)
    cosine = paired_cosine(raw_activations, official_activations)

    np.savez_compressed(
        args.output_dir / "paired_layer_activations.npz",
        rows=rows,
        raw=raw_activations,
        official=official_activations,
        cka=matrix,
    )
    report = {
        "status": "pipeline_smoke_test_only",
        "warning": (
            "The tiny pilot validates extraction and comparison only. Do not "
            "select layers or claim chemical factors from these values."
        ),
        "data": str(args.data.resolve()),
        "fold": args.fold,
        "n_spectra": len(rows),
        "seed": args.seed,
        "n_highest_peaks": args.n_highest_peaks,
        "raw_checkpoint": str(args.raw_checkpoint.resolve()),
        "official_checkpoint": str(args.official_checkpoint.resolve()),
        "raw_diagnostics": raw_diagnostics,
        "official_diagnostics": official_diagnostics,
        "same_layer_cka": diagonal.tolist(),
        "paired_same_layer_cosine": cosine,
        "cka_matrix": matrix.tolist(),
        "metadata": metadata,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# DreaMS 原始—官方微调模型配对层表征：冒烟测试",
        "",
        "> 该结果只验证管线，样本过小，不能据此选择层或宣称发现了化学因子。",
        "",
        f"- 同一批谱图：{len(rows)} 条（fold={args.fold}）",
        "- 层表征定义：每个完整 Transformer block 后的 precursor token；",
        "  最后一层使用主干实际返回的最终归一化表征。",
        "- 原始模型重复前向最大绝对差：" + (
            f"{raw_diagnostics['determinism_max_abs']:.3g}"
            if raw_diagnostics["determinism_max_abs"] is not None else "未执行"
        ),
        "- 官方模型重复前向最大绝对差：" + (
            f"{official_diagnostics['determinism_max_abs']:.3g}"
            if official_diagnostics["determinism_max_abs"] is not None else "未执行"
        ),
        "",
        "| 层 | 同层 CKA | 同谱图向量平均余弦 |",
        "|---:|---:|---:|",
    ]
    for layer, (cka_value, cosine_value) in enumerate(zip(diagonal, cosine), start=1):
        lines.append(f"| {layer} | {cka_value:.6f} | {cosine_value:.6f} |")
    lines.extend([
        "",
        "## 当前只允许得出的结论",
        "",
        "1. 两套同构主干能够在相同预处理、相同谱图顺序下逐层配对。",
        "2. 捕获的是完整 block 输出，不是残差相加前的前馈分支。",
        "3. 数值全部有限且重复前向一致后，才允许扩到 1,000 条做稳定性测试。",
        "4. 30,000--50,000 条平衡样本完成后，才根据 CKA 选择 Crosscoder 层。",
    ])
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print("[5/5] Saved report", flush=True)
    print("  same-layer CKA:", " ".join(f"{v:.4f}" for v in diagonal), flush=True)
    print(f"  {args.output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
