"""Dependency-minimal frozen-prefix cache for direct DreaMS fine-tuning.

This module intentionally depends only on NumPy and PyTorch.  In particular,
it must not import the historical ChemAware PEFT/probe stack: direct action
training uses the cache as a computation accelerator, not as a PEFT model.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F


class FrozenPrefixSpectrumStore:
    """Cache the frozen DreaMS prefix and rerun every trainable final block."""

    def __init__(
        self,
        model: torch.nn.Module,
        source,
        device: torch.device,
        batch_size: int,
        last_blocks: int = 1,
    ):
        if batch_size < 1:
            raise ValueError("prefix cache batch size must be positive")
        backbone = model.backbone
        encoder = backbone.transformer_encoder
        if not getattr(encoder, "pre_norm", False):
            raise RuntimeError("frozen-prefix cache currently requires pre-norm DreaMS")
        final_layer = int(encoder.n_layers) - 1
        if not 1 <= last_blocks < int(encoder.n_layers):
            raise ValueError("prefix cache requires between one and n_layers-1 final blocks")
        start_layer = int(encoder.n_layers) - int(last_blocks)
        if start_layer < 1 or not getattr(backbone, "d_fourier", 0):
            raise RuntimeError("unsupported DreaMS prefix-cache architecture")
        adapted_layers = tuple(range(start_layer, final_layer + 1))
        if any(getattr(encoder.atts[layer], "d_graphormer_params", 0) for layer in adapted_layers):
            raise RuntimeError(
                "prefix cache currently supports the official scalar Graphormer path only"
            )

        self.rows = np.asarray(source.rows, dtype=np.int64).copy()
        self.position = dict(source.position)
        if (
            self.rows.ndim != 1 or len(self.rows) == 0
            or len(self.position) != len(self.rows)
            or set(map(int, self.rows)) != set(map(int, self.position))
            or set(map(int, self.position.values())) != set(range(len(self.rows)))
        ):
            raise RuntimeError("prefix-cache source rows are not a unique aligned vector")
        prefix_chunks = []
        projected_chunks = []
        mask_chunks = []
        model.eval()
        dtype = next(model.parameters()).dtype
        captured: list[torch.Tensor] = []

        def capture_prefix(_module, inputs):
            captured.append(inputs[0].detach())

        handle = encoder.scales[2 * start_layer].register_forward_pre_hook(capture_prefix)
        try:
            with torch.no_grad():
                cache_started = time.monotonic()
                last_progress = cache_started
                print(
                    f"[prefix-cache] start source={type(source).__name__} "
                    f"spectra={len(self.rows)} batch_size={batch_size}",
                    flush=True,
                )
                for left in range(0, len(self.rows), batch_size):
                    rows = self.rows[left:left + batch_size]
                    spectra = source.get(rows).to(device=device, dtype=dtype)
                    captured.clear()
                    model.backbone(spectra, None)
                    if len(captured) != 1:
                        raise RuntimeError("failed to capture exactly one trainable-prefix input")
                    prefix_chunks.append(captured[0].float().cpu())
                    mask_chunks.append((spectra[:, :, 0] == 0).cpu())
                    fourier = backbone.ff_fourier(backbone.fourier_enc(spectra[..., [0]]))
                    projected = torch.sum(fourier, dim=-1, keepdim=True)
                    # Store the exact per-token scalar and materialize the
                    # pairwise difference only for the requested mini-batch.
                    # This reduces cache memory from O(N*L^2) to O(N*L).
                    projected_chunks.append(projected.float().cpu())
                    now = time.monotonic()
                    completed = min(left + len(rows), len(self.rows))
                    if now - last_progress >= 30 or completed == len(self.rows):
                        elapsed = now - cache_started
                        rate = completed / max(elapsed, 1e-9)
                        remaining = (len(self.rows) - completed) / max(rate, 1e-9)
                        print(
                            f"[prefix-cache] progress source={type(source).__name__} "
                            f"{completed}/{len(self.rows)} elapsed={elapsed:.1f}s "
                            f"eta={remaining:.1f}s",
                            flush=True,
                        )
                        last_progress = now
        finally:
            handle.remove()
        self.prefix = torch.cat(prefix_chunks, dim=0).contiguous()
        self.graphormer_projected = torch.cat(projected_chunks, dim=0).contiguous()
        self.padding_mask = torch.cat(mask_chunks, dim=0).contiguous()
        self.start_layer = start_layer
        self.final_layer = final_layer
        self.adapted_layers = adapted_layers
        self.audit = {
            "enabled": True,
            "implementation": "dependency_minimal_direct_cache_v1",
            "cached_spectra": int(len(self.rows)),
            "start_layer": start_layer,
            "final_layer": final_layer,
            "adapted_layers": list(adapted_layers),
            "prefix_shape": list(self.prefix.shape),
            "graphormer_projected_shape": list(self.graphormer_projected.shape),
            "prefix_bytes": int(self.prefix.numel() * self.prefix.element_size()),
            "graphormer_projected_bytes": int(
                self.graphormer_projected.numel() * self.graphormer_projected.element_size()
            ),
            "graphormer_bias_materialized_per_batch": True,
            "training_only_computation_cache": True,
        }

    def forward(
        self,
        model: torch.nn.Module,
        rows: np.ndarray,
        device: torch.device,
        batch_size: int,
        amp: bool,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("prefix-cache forward batch size must be positive")
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or len(rows) == 0:
            raise ValueError("prefix-cache forward rows must be a non-empty vector")
        try:
            positions = np.asarray(
                [self.position[int(row)] for row in rows],
                dtype=np.int64,
            )
        except KeyError as error:
            raise RuntimeError(f"spectrum row absent from prefix cache: {error}") from error
        outputs = []
        encoder = model.backbone.transformer_encoder
        dtype = next(model.parameters()).dtype
        for left in range(0, len(positions), batch_size):
            index = positions[left:left + batch_size]
            x = self.prefix[index].to(device=device, dtype=dtype)
            mask = self.padding_mask[index].to(device=device)
            projected = self.graphormer_projected[index].to(device=device, dtype=dtype)
            bias = projected.unsqueeze(2) - projected.unsqueeze(1)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(amp and device.type == "cuda"),
            ):
                for layer in self.adapted_layers:
                    x = encoder._layer_forward(layer, x, mask, bias)
                x = encoder.scales[-1](x)
                outputs.append(F.normalize(model.head(x[:, 0]), dim=-1))
        output = torch.cat(outputs, dim=0)
        if not bool(torch.all(torch.isfinite(output))):
            raise RuntimeError("prefix-cache suffix produced non-finite embeddings")
        norms = torch.linalg.vector_norm(output.float(), dim=1)
        if not bool(torch.all(torch.abs(norms - 1.0) <= 2e-3)):
            raise RuntimeError("prefix-cache suffix produced non-unit embeddings")
        return output
