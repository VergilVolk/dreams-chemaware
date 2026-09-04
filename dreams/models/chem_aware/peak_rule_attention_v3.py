"""Training-only peak-rule attention views for shared DreaMS PEFT.

``binary_union`` retains the original all-edge diagnostic. ``idf_precursor``
uses only relations that directly connect the precursor query to a fragment:
precursor-to-fragment neutral losses and characteristic fragment masses.
Targets are weighted by corpus inverse document frequency and aliases are
combined by a maximum rather than double-counted.

Every view is candidate-independent and discarded at inference. The stronger
``peak_permuted`` control keeps each spectrum's exact bias values, sparsity,
valid-peak count and row degree, but rotates them across its fragment tokens.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np
import torch

from dreams.models.chem_aware.chem_rules import ChemicalRuleEngine


class _RawSpectrumSource(Protocol):
    rows: np.ndarray
    tensor: torch.Tensor


def deterministic_spectrum_permutation(n_spectra: int, seed: int) -> np.ndarray:
    """Return a deterministic derangement that preserves every bias matrix."""

    if n_spectra < 2:
        raise ValueError("a spectrum-permuted control requires at least two spectra")
    shift = int(seed) % (int(n_spectra) - 1) + 1
    return np.roll(np.arange(n_spectra, dtype=np.int64), shift)


def deterministic_peak_permutation(n_fragments: int, seed: int) -> np.ndarray:
    """Return a cyclic derangement of fragment positions (precursor excluded)."""

    if n_fragments < 2:
        return np.arange(n_fragments, dtype=np.int64)
    shift = int(seed) % (int(n_fragments) - 1) + 1
    return np.roll(np.arange(n_fragments, dtype=np.int64), shift)


def _sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest()


def _inverse_document_weights(matches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return capped [0.1, 1] IDF weights and document frequencies."""

    if matches.ndim != 3:
        raise ValueError("rule matches must have shape (spectra, rules, fragments)")
    document_frequency = matches.any(dim=-1).sum(dim=0).float()
    n_spectra = int(matches.shape[0])
    raw = torch.log(torch.tensor(float(n_spectra + 1)) / (document_frequency + 1.0))
    positive = raw[raw > 0]
    if len(positive) == 0:
        return torch.zeros_like(raw), document_frequency
    normalizer = torch.quantile(positive, 0.9).clamp_min(1e-8)
    weights = (raw / normalizer).clamp(min=0.1, max=1.0)
    weights = torch.where(document_frequency > 0, weights, torch.zeros_like(weights))
    return weights, document_frequency


def _frequency_summary(prefix: str, frequency: torch.Tensor) -> dict:
    observed = frequency[frequency > 0]
    return {
        f"{prefix}_targets": int(len(frequency)),
        f"{prefix}_targets_observed": int(len(observed)),
        f"{prefix}_document_frequency_min": int(observed.min()) if len(observed) else 0,
        f"{prefix}_document_frequency_median": float(torch.median(observed)) if len(observed) else 0.0,
        f"{prefix}_document_frequency_max": int(observed.max()) if len(observed) else 0,
    }


def _idf_precursor_bias(
    spectra: torch.Tensor,
    engine: ChemicalRuleEngine,
    categories: tuple[str, ...],
) -> tuple[torch.Tensor, dict]:
    """Build a precursor-query-only, corpus-specific chemical attention prior."""

    spectra = spectra.float().cpu()
    mz = spectra[..., 0]
    valid = mz != 0
    fragments = mz[:, 1:]
    valid_fragments = valid[:, 1:]
    precursor = mz[:, [0]]
    n_spectra, n_tokens = mz.shape
    score = torch.zeros(n_spectra, n_tokens - 1, dtype=torch.float32)
    audit: dict[str, object] = {
        "precursor_query_only": True,
        "idf_quantile_normalization": 0.9,
        "rule_alias_aggregation": "maximum_per_fragment",
    }

    if "NL" in categories and len(engine.md_targets) > 0:
        targets = engine.md_targets.float().cpu()
        losses = torch.abs(precursor - fragments)
        matches = (
            torch.abs(losses.unsqueeze(1) - targets.view(1, -1, 1))
            < float(engine.tolerance)
        ) & valid_fragments.unsqueeze(1)
        weights, frequency = _inverse_document_weights(matches)
        score = torch.maximum(
            score,
            (matches.float() * weights.view(1, -1, 1)).amax(dim=1),
        )
        audit.update(_frequency_summary("nl", frequency))

    if "CF" in categories and len(engine.pm_targets) > 0:
        targets = engine.pm_targets.float().cpu()
        matches = (
            torch.abs(fragments.unsqueeze(1) - targets.view(1, -1, 1))
            < float(engine.tolerance)
        ) & valid_fragments.unsqueeze(1)
        weights, frequency = _inverse_document_weights(matches)
        score = torch.maximum(
            score,
            (matches.float() * weights.view(1, -1, 1)).amax(dim=1),
        )
        audit.update(_frequency_summary("cf", frequency))

    maximum = score.amax(dim=1, keepdim=True)
    score = torch.where(maximum > 0, score / maximum.clamp_min(1e-8), score)
    score = score * valid_fragments.float()
    bias = torch.zeros(n_spectra, 1, n_tokens, n_tokens, dtype=torch.float32)
    bias[:, 0, 0, 1:] = score
    audit["spectra_with_any_bias"] = int(torch.count_nonzero(score.amax(dim=1)))
    audit["matched_fragments"] = int(torch.count_nonzero(score))
    audit["matched_fragments_per_spectrum_median"] = float(
        torch.median(torch.count_nonzero(score, dim=1).float())
    )
    return bias, audit


class PeakRuleBiasStore:
    """Cache fixed peak-rule attention priors for graph-reachable spectra."""

    def __init__(
        self,
        source: _RawSpectrumSource,
        *,
        scale: float,
        control: str,
        seed: int,
        categories: tuple[str, ...] = ("NL", "CF", "ISO"),
        bias_kind: str = "binary_union",
        batch_size: int = 4,
    ):
        if scale <= 0:
            raise ValueError("peak-rule attention scale must be positive")
        if control not in {"correct", "spectrum_permuted", "peak_permuted"}:
            raise ValueError("unknown peak-rule control")
        if bias_kind not in {"binary_union", "idf_precursor"}:
            raise ValueError("unknown peak-rule bias kind")
        if batch_size < 1:
            raise ValueError("peak-rule cache batch size must be positive")
        if not categories or not set(categories) <= {"NL", "CF", "ISO"}:
            raise ValueError("peak-rule categories must be a nonempty NL/CF/ISO subset")
        if bias_kind == "idf_precursor" and not set(categories) <= {"NL", "CF"}:
            raise ValueError("idf_precursor currently supports NL and CF only")

        self.rows = np.asarray(source.rows, dtype=np.int64).copy()
        self.position = {int(row): index for index, row in enumerate(self.rows)}
        if len(self.position) != len(self.rows):
            raise RuntimeError("peak-rule rows must be unique")
        self.scale = float(scale)
        self.control = str(control)
        self.categories = tuple(categories)
        self.bias_kind = str(bias_kind)

        engine = ChemicalRuleEngine(
            tolerance=0.02,
            enable_categories=list(categories),
            use_massbank=False,
        ).cpu().eval()
        engine._debug_done = 3
        if bias_kind == "binary_union":
            chunks = []
            with torch.no_grad():
                for left in range(0, len(self.rows), batch_size):
                    spectra = source.tensor[left:left + batch_size].float().cpu()
                    mz = spectra[..., 0]
                    padding = mz == 0
                    bias = engine(
                        ChemicalRuleEngine.compute_peak_pair_mz_diffs(mz),
                        mz_values=mz,
                        precursor_mz=mz[:, 0],
                        padding_mask=padding,
                        categories=list(categories),
                    )
                    binary = bias > 0
                    diagonal = torch.eye(
                        binary.shape[-1], dtype=torch.bool
                    ).view(1, 1, binary.shape[-1], binary.shape[-1])
                    chunks.append((binary & ~diagonal).float().cpu())
            self.base_bias = torch.cat(chunks, dim=0).contiguous()
            construction_audit: dict[str, object] = {"legacy_binary_union": True}
        else:
            self.base_bias, construction_audit = _idf_precursor_bias(
                source.tensor, engine, self.categories
            )
            self.base_bias = self.base_bias.contiguous()

        if self.base_bias.shape[0] != len(self.rows):
            raise RuntimeError("peak-rule cache row count mismatch")

        self.source_index = np.arange(len(self.rows), dtype=np.int64)
        self.permuted_bias: torch.Tensor | None = None
        peak_permutation_fixed_points = 0
        peak_permutation_eligible_spectra = 0
        if control == "spectrum_permuted":
            self.source_index = deterministic_spectrum_permutation(len(self.rows), seed)
            if np.any(self.source_index == np.arange(len(self.rows))):
                raise RuntimeError("spectrum-permuted control contains fixed points")
        elif control == "peak_permuted":
            self.permuted_bias = self.base_bias.clone()
            valid_counts = torch.count_nonzero(source.tensor[..., 0], dim=1).numpy()
            for index, valid_count in enumerate(valid_counts):
                n_fragments = int(valid_count) - 1
                if n_fragments < 2:
                    continue
                peak_permutation_eligible_spectra += 1
                permutation = deterministic_peak_permutation(
                    n_fragments, seed + 104729 * int(self.rows[index])
                )
                peak_permutation_fixed_points += int(np.sum(
                    permutation == np.arange(n_fragments)
                ))
                source_columns = torch.arange(1, n_fragments + 1)
                target_columns = torch.as_tensor(permutation + 1, dtype=torch.long)
                self.permuted_bias[index, :, :, target_columns] = self.base_bias[
                    index, :, :, source_columns
                ]
            if peak_permutation_fixed_points:
                raise RuntimeError("peak-permuted control contains fragment fixed points")
            original_values = torch.sort(
                self.base_bias.reshape(len(self.rows), -1), dim=1
            ).values
            permuted_values = torch.sort(
                self.permuted_bias.reshape(len(self.rows), -1), dim=1
            ).values
            if not torch.equal(original_values, permuted_values):
                raise RuntimeError("peak-permuted control changed bias values")

        counts = torch.count_nonzero(
            self.base_bias.reshape(len(self.rows), -1), dim=1
        ).numpy()
        mapped_counts = counts[self.source_index]
        if not np.array_equal(np.sort(mapped_counts), np.sort(counts)):
            raise RuntimeError("peak-rule control failed marginal-density preservation")
        effective = (
            self.permuted_bias
            if self.permuted_bias is not None
            else self.base_bias[self.source_index]
        )
        self.audit = {
            "kind": "training_only_peak_rule_attention_view",
            "bias_kind": bias_kind,
            "control": control,
            "scale": float(scale),
            "categories": list(categories),
            "rule_count": int(len(engine.rules)),
            "spectra": int(len(self.rows)),
            "matrix_shape": list(self.base_bias.shape),
            "nonzero_edges": int(torch.count_nonzero(self.base_bias)),
            "nonzero_edges_per_spectrum_min": int(counts.min()),
            "nonzero_edges_per_spectrum_median": float(np.median(counts)),
            "nonzero_edges_per_spectrum_max": int(counts.max()),
            "bias_sum": float(self.base_bias.sum()),
            "bias_max": float(self.base_bias.max()),
            "spectrum_alignment_preserved": control == "correct",
            "within_spectrum_topology_preserved": control in {"correct", "peak_permuted"},
            "marginal_bias_collection_preserved": True,
            "fixed_points": int(np.sum(
                self.source_index == np.arange(len(self.rows))
            )) if control != "peak_permuted" else 0,
            "peak_permutation_eligible_spectra": peak_permutation_eligible_spectra,
            "peak_permutation_fixed_points": peak_permutation_fixed_points,
            "row_ledger_sha256": hashlib.sha256(
                np.ascontiguousarray(self.rows).tobytes()
            ).hexdigest(),
            "source_index_sha256": hashlib.sha256(
                np.ascontiguousarray(self.source_index).tobytes()
            ).hexdigest(),
            "base_bias_sha256": _sha256_tensor(self.base_bias),
            "effective_bias_sha256": _sha256_tensor(effective),
            "construction": construction_audit,
            "candidate_inputs_used": False,
            "molecule_identity_used": False,
            "discarded_at_inference": True,
        }

    def get(
        self,
        rows: np.ndarray,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        try:
            position = np.asarray(
                [self.position[int(row)] for row in np.asarray(rows, dtype=np.int64)],
                dtype=np.int64,
            )
        except KeyError as error:
            raise RuntimeError(f"spectrum row absent from peak-rule cache: {error}") from error
        if self.permuted_bias is not None:
            value = self.permuted_bias[position]
        else:
            value = self.base_bias[self.source_index[position]]
        return value.to(device=device, dtype=dtype) * self.scale
