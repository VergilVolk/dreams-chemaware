"""Audit a deployable mass-conservation kernel appended to official DreaMS.

Each spectrum is independently mapped to a fixed-size feature vector from its
fragment m/z and precursor-minus-fragment neutral losses.  Concatenation with
the unit official embedding gives a genuine shared embedding whose dot product
is the weighted sum of official and mass-kernel similarities.  No candidate-set
normalization, candidate structure, or query-specific scoring function is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # NumPy-only action scoring remains available.
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_numpy_sampling import (  # noqa: E402
    formula_bootstrap, identity_balanced_queries, official_outcomes,
    stable_formula_folds,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


DEFAULT_VARIANTS = ("mass", "fragment", "uniform_mass", "intensity_permuted", "mass_shifted")
PAIR_VARIANTS = ("pair_gap", "mass_pair", "mass_pair_intensity_permuted", "mass_pair_mz_jittered")
MULTISCALE_VARIANTS = ("multiscale_mass", "multiscale_intensity_permuted", "multiscale_mass_shifted")
DUAL_VARIANTS = ("mass_uniform", "mass_uniform_intensity_permuted", "mass_uniform_mass_shifted")
RULE_VARIANTS = (
    "rule_response", "rule_mass", "rule_mass_shifted",
    "rule_response_shifted", "rule_response_row_permuted",
    "rule_response_content_permuted", "rule_response_content_permuted_b",
    "rule_response_content_permuted_c", "rule_response_local_background_a",
    "rule_response_local_background_b", "rule_response_local_background_c",
    "rule_response_centered_local", "rule_response_centered_positive",
    "rule_response_local_contrast_a", "rule_response_local_contrast_b",
    "rule_response_local_contrast_c",
    "rule_mass_row_permuted",
)
VARIANTS = DEFAULT_VARIANTS + PAIR_VARIANTS + MULTISCALE_VARIANTS + DUAL_VARIANTS + RULE_VARIANTS


def content_permutation_offset(
    mz: np.ndarray, intensity: np.ndarray, precursor: float, dimension: int,
    salt: bytes = b"",
) -> int:
    """Return an avalanche-hashed rotation that is independent of dataset row IDs."""
    if dimension <= 1:
        return 0
    digest = hashlib.sha256()
    digest.update(np.asarray(mz, dtype="<f4").tobytes())
    digest.update(np.asarray(intensity, dtype="<f4").tobytes())
    digest.update(np.asarray([precursor], dtype="<f4").tobytes())
    digest.update(bytes(salt))
    return 1 + int.from_bytes(digest.digest()[:8], "little") % (dimension - 1)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adapter-checkpoint", type=Path, default=None,
        help="Optional passed shared residual adapter to use as the global embedding base.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--discovery-errors", type=int, default=256)
    parser.add_argument("--discovery-correct", type=int, default=256)
    parser.add_argument(
        "--discovery-natural-identities", type=int, default=2048,
        help="Identity-equal training-formula sample with natural error prevalence used for beta selection.",
    )
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--multi-bin-widths", type=float, nargs="+", default=(0.01, 0.02, 0.05))
    parser.add_argument("--uniform-channel-weight", type=float, default=1.0)
    parser.add_argument("--rule-library", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--rule-channel-weight", type=float, default=1.0)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=DEFAULT_VARIANTS)
    parser.add_argument("--primary-variant", choices=VARIANTS, default="mass")
    parser.add_argument(
        "--control-variants", nargs="+", choices=VARIANTS,
        default=("fragment", "uniform_mass", "intensity_permuted", "mass_shifted"),
    )
    parser.add_argument(
        "--required-control-variants", nargs="+", choices=VARIANTS, default=None,
        help="Subset of reported controls that define admission; defaults to all controls.",
    )
    parser.add_argument("--fusion-grid", type=float, nargs="+", default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60))
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def strict_rank(scores: np.ndarray, labels: np.ndarray) -> int:
    positive = float(scores[np.flatnonzero(labels)[0]])
    return 1 + int(np.sum(scores[~labels] >= positive))


def metric(old_rank: np.ndarray, new_rank: np.ndarray) -> dict:
    old_hit = old_rank == 1; new_hit = new_rank == 1
    return {
        "queries": int(len(old_rank)), "baseline_recall1": float(np.mean(old_hit)),
        "recall1": float(np.mean(new_hit)), "delta_recall1": float(np.mean(new_hit) - np.mean(old_hit)),
        "baseline_mrr": float(np.mean(1 / old_rank)), "mrr": float(np.mean(1 / new_rank)),
        "delta_mrr": float(np.mean(1 / new_rank) - np.mean(1 / old_rank)),
        "corrected": int(np.sum(~old_hit & new_hit)), "introduced": int(np.sum(old_hit & ~new_hit)),
    }


def hash_index(key: int, dimension: int) -> tuple[int, float]:
    value = (int(key) * 2654435761 + 2246822519) & 0xFFFFFFFF
    index = value % dimension
    sign_value = (value * 3266489917 + 668265263) & 0xFFFFFFFF
    return int(index), (1.0 if sign_value & 1 else -1.0)


class KernelCache:
    def __init__(
        self, args: argparse.Namespace, row_position: dict[int, int],
        variants: tuple[str, ...] = DEFAULT_VARIANTS,
    ):
        unknown = set(variants) - set(VARIANTS)
        if unknown:
            raise ValueError(f"unknown mass-kernel variants: {sorted(unknown)}")
        self.args = args; self.row_position = row_position
        self.variants = tuple(variants)
        self.mz = np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")
        self.intensity = np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r")
        self.valid = np.load(args.token_dir / "valid.npy", mmap_mode="r")
        self.precursor = np.load(args.token_dir / "precursor_mz_f32.npy", mmap_mode="r")
        rule_payload = json.loads(args.rule_library.read_text(encoding="utf-8"))
        if "channels" in rule_payload:
            rule_records = [
                {
                    "category": channel["category"],
                    "match_type": channel["match_type"],
                    "value": channel["value_da"],
                }
                for channel in rule_payload["channels"]
            ]
        else:
            rule_records = rule_payload["rules"]
        self.nl_rules = np.asarray([
            float(rule["value"]) for rule in rule_records
            if rule.get("category") == "NL" and rule.get("match_type") == "mass_diff"
        ], dtype=np.float64)
        self.cf_rules = np.asarray([
            float(rule["value"]) for rule in rule_records
            if rule.get("category") == "CF" and rule.get("match_type") == "peak_mz"
        ], dtype=np.float64)
        self.cache: dict[int, dict[str, np.ndarray]] = {}

    def _vector(
        self, mz: np.ndarray, weight: np.ndarray, precursor: float,
        channels: tuple[str, ...], bin_width: float | None = None,
    ) -> np.ndarray:
        out = np.zeros(self.args.kernel_dim, dtype=np.float32)
        width = self.args.bin_width if bin_width is None else float(bin_width)
        offsets = np.arange(self.args.grid_offsets, dtype=np.float64) / self.args.grid_offsets
        for channel_index, channel in enumerate(channels):
            if channel == "fragment":
                value = mz; channel_weight = weight
            elif channel == "neutral_loss":
                value = precursor - mz; channel_weight = weight
            elif channel == "pair_gap":
                left, right = np.triu_indices(len(mz), k=1)
                value = np.abs(mz[left] - mz[right])
                channel_weight = self.args.pair_weight * weight[left] * weight[right]
            else:
                raise ValueError(f"unknown mass channel: {channel}")
            for grid_index, offset in enumerate(offsets):
                bins = np.floor(value / width + offset).astype(np.int64)
                namespace = channel_index * 1_000_000_007 + grid_index * 100_000_3
                for mass_bin, amplitude in zip(bins, channel_weight):
                    index, sign = hash_index(namespace + int(mass_bin), self.args.kernel_dim)
                    out[index] += sign * float(amplitude)
        norm = float(np.linalg.norm(out))
        if norm:
            out /= norm
        return out.astype(np.float16)

    def _multiscale(
        self, mz: np.ndarray, weight: np.ndarray, precursor: float,
    ) -> np.ndarray:
        parts = [
            self._vector(mz, weight, precursor, ("fragment", "neutral_loss"), width)
            for width in self.args.multi_bin_widths
        ]
        if not parts:
            raise ValueError("multi-bin-widths cannot be empty")
        return (np.concatenate(parts).astype(np.float32) / np.sqrt(len(parts))).astype(np.float16)

    def _mass_uniform(
        self, mz: np.ndarray, weighted: np.ndarray, uniform: np.ndarray,
        precursor: float,
    ) -> np.ndarray:
        mass = self._vector(mz, weighted, precursor, ("fragment", "neutral_loss")).astype(np.float32)
        presence = self._vector(mz, uniform, precursor, ("fragment", "neutral_loss")).astype(np.float32)
        scale = float(self.args.uniform_channel_weight)
        return (np.concatenate((mass, scale * presence)) / np.sqrt(1.0 + scale * scale)).astype(np.float16)

    def _rule_response(
        self, mz: np.ndarray, weight: np.ndarray, precursor: float,
        observed_shift: float = 0.0,
    ) -> np.ndarray:
        shifted = mz + observed_shift
        loss = precursor - shifted
        tolerance = float(self.args.rule_tolerance)
        def responses(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
            if not len(values) or not len(targets):
                return np.zeros(len(targets), dtype=np.float32)
            distance = np.abs(targets[:, None] - values[None, :])
            affinity = np.maximum(0.0, 1.0 - distance / tolerance)
            return np.max(affinity * weight[None, :], axis=1).astype(np.float32)
        out = np.concatenate((
            responses(loss, self.nl_rules), responses(shifted, self.cf_rules),
        ))
        norm = float(np.linalg.norm(out))
        if norm:
            out /= norm
        return out.astype(np.float16)

    def _rule_response_local_background(
        self, mz: np.ndarray, weight: np.ndarray, precursor: float,
        offset: float,
    ) -> np.ndarray:
        """Matched shared-coordinate null at masses flanking each rule target."""
        loss = precursor - mz
        tolerance = float(self.args.rule_tolerance)

        def responses(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
            if not len(values) or not len(targets):
                return np.zeros(len(targets), dtype=np.float32)
            distance = np.abs(targets[:, None] - values[None, :])
            affinity = np.maximum(0.0, 1.0 - distance / tolerance)
            return np.max(affinity * weight[None, :], axis=1).astype(np.float32)

        sides = []
        for shift in (-float(offset), float(offset)):
            sides.append(np.concatenate((
                responses(loss, self.nl_rules + shift),
                responses(mz, self.cf_rules + shift),
            )))
        out = 0.5 * (sides[0] + sides[1])
        norm = float(np.linalg.norm(out))
        if norm:
            out /= norm
        return out.astype(np.float16)

    def _rule_mass(
        self, mz: np.ndarray, weight: np.ndarray, precursor: float,
        rule_shift: float = 0.0,
    ) -> np.ndarray:
        mass = self._vector(mz, weight, precursor, ("fragment", "neutral_loss")).astype(np.float32)
        rules = self._rule_response(mz, weight, precursor, rule_shift).astype(np.float32)
        scale = float(self.args.rule_channel_weight)
        return (np.concatenate((mass, scale * rules)) / np.sqrt(1.0 + scale * scale)).astype(np.float16)

    def get(self, row: int) -> dict[str, np.ndarray]:
        row = int(row)
        if row in self.cache:
            return self.cache[row]
        pos = self.row_position[row]
        available = np.flatnonzero(self.valid[pos])
        order = available[np.argsort(-np.asarray(self.intensity[pos, available]), kind="stable")[:self.args.top_peaks]]
        mz = np.asarray(self.mz[pos, order], dtype=np.float64)
        intensity = np.maximum(np.asarray(self.intensity[pos, order], dtype=np.float64), 0)
        intensity /= max(float(np.sum(intensity)), 1e-12)
        weighted = np.power(intensity, self.args.intensity_power)
        uniform = np.ones(len(mz), dtype=np.float64)
        if len(uniform):
            uniform /= np.sqrt(len(uniform))
        permuted = weighted.copy()
        if len(permuted) > 1:
            permuted = np.roll(permuted, 1 + row % (len(permuted) - 1))
        shift_direction = 1.0 if row % 2 else -1.0
        shifted_mz = mz + shift_direction * self.args.mass_shift_da
        alternating = np.where(np.arange(len(mz)) % 2, 1.0, -1.0)
        jittered_mz = mz + alternating * self.args.mass_shift_da
        precursor = float(self.precursor[pos])
        rule_dimension = len(self.nl_rules) + len(self.cf_rules)
        content_rule_offset = content_permutation_offset(
            mz, intensity, precursor, rule_dimension,
        )
        content_rule_offset_b = content_permutation_offset(
            mz, intensity, precursor, rule_dimension, b"chemaware-null-b",
        )
        content_rule_offset_c = content_permutation_offset(
            mz, intensity, precursor, rule_dimension, b"chemaware-null-c",
        )
        local_rule_cache: dict[str, np.ndarray] = {}

        def local_rule_centers() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            if not local_rule_cache:
                local_rule_cache["raw"] = self._rule_response(
                    mz, weighted, precursor,
                ).astype(np.float32)
                for name, offset in (("a", 0.071), ("b", 0.137), ("c", 0.223)):
                    local_rule_cache[name] = self._rule_response_local_background(
                        mz, weighted, precursor, offset,
                    ).astype(np.float32)
            return (
                local_rule_cache["raw"], local_rule_cache["a"],
                local_rule_cache["b"], local_rule_cache["c"],
            )

        def normalized_contrast(center: np.ndarray, others: tuple[np.ndarray, ...], positive: bool = False) -> np.ndarray:
            value = center - np.mean(np.stack(others), axis=0)
            if positive:
                value = np.maximum(value, 0.0)
            norm = float(np.linalg.norm(value))
            if norm:
                value = value / norm
            return value.astype(np.float16)
        builders = {
            "mass": lambda: self._vector(mz, weighted, precursor, ("fragment", "neutral_loss")),
            "fragment": lambda: self._vector(mz, weighted, precursor, ("fragment",)),
            "uniform_mass": lambda: self._vector(mz, uniform, precursor, ("fragment", "neutral_loss")),
            "intensity_permuted": lambda: self._vector(mz, permuted, precursor, ("fragment", "neutral_loss")),
            "mass_shifted": lambda: self._vector(shifted_mz, weighted, precursor, ("fragment", "neutral_loss")),
            "pair_gap": lambda: self._vector(mz, weighted, precursor, ("pair_gap",)),
            "mass_pair": lambda: self._vector(mz, weighted, precursor, ("fragment", "neutral_loss", "pair_gap")),
            "mass_pair_intensity_permuted": lambda: self._vector(
                mz, permuted, precursor, ("fragment", "neutral_loss", "pair_gap")
            ),
            "mass_pair_mz_jittered": lambda: self._vector(
                jittered_mz, weighted, precursor, ("fragment", "neutral_loss", "pair_gap")
            ),
            "multiscale_mass": lambda: self._multiscale(mz, weighted, precursor),
            "multiscale_intensity_permuted": lambda: self._multiscale(mz, permuted, precursor),
            "multiscale_mass_shifted": lambda: self._multiscale(shifted_mz, weighted, precursor),
            "mass_uniform": lambda: self._mass_uniform(mz, weighted, uniform, precursor),
            "mass_uniform_intensity_permuted": lambda: self._mass_uniform(mz, permuted, uniform, precursor),
            "mass_uniform_mass_shifted": lambda: self._mass_uniform(shifted_mz, weighted, uniform, precursor),
            "rule_response": lambda: self._rule_response(mz, weighted, precursor),
            "rule_mass": lambda: self._rule_mass(mz, weighted, precursor),
            "rule_mass_shifted": lambda: self._rule_mass(
                mz, weighted, precursor, shift_direction * self.args.mass_shift_da
            ),
            "rule_response_shifted": lambda: self._rule_response(
                mz, weighted, precursor, shift_direction * self.args.mass_shift_da
            ),
            # Spectrum-specific cyclic permutations preserve response norms,
            # sparsity and compute while destroying consistent rule identity
            # across query/reference spectra: a matched semantics-null arm.
            "rule_response_row_permuted": lambda: np.roll(
                self._rule_response(mz, weighted, precursor),
                1 + row % max(1, len(self.nl_rules) + len(self.cf_rules) - 1),
            ),
            # Semantics-null matched control without the row-order leakage of
            # ``rule_response_row_permuted``.  The offset is an avalanche hash
            # of the clean spectrum itself, never of its library row/source.
            "rule_response_content_permuted": lambda: np.roll(
                self._rule_response(mz, weighted, precursor), content_rule_offset,
            ),
            "rule_response_content_permuted_b": lambda: np.roll(
                self._rule_response(mz, weighted, precursor), content_rule_offset_b,
            ),
            "rule_response_content_permuted_c": lambda: np.roll(
                self._rule_response(mz, weighted, precursor), content_rule_offset_c,
            ),
            # Shared-coordinate matched nulls: every spectrum uses the same
            # flanking target masses.  They preserve channel count and global
            # coordinate consistency while removing the exact curated masses.
            "rule_response_local_background_a": lambda: self._rule_response_local_background(
                mz, weighted, precursor, 0.071,
            ),
            "rule_response_local_background_b": lambda: self._rule_response_local_background(
                mz, weighted, precursor, 0.137,
            ),
            "rule_response_local_background_c": lambda: self._rule_response_local_background(
                mz, weighted, precursor, 0.223,
            ),
            "rule_response_centered_local": lambda: normalized_contrast(
                local_rule_centers()[0], local_rule_centers()[1:],
            ),
            "rule_response_centered_positive": lambda: normalized_contrast(
                local_rule_centers()[0], local_rule_centers()[1:], positive=True,
            ),
            "rule_response_local_contrast_a": lambda: normalized_contrast(
                local_rule_centers()[1], local_rule_centers()[2:],
            ),
            "rule_response_local_contrast_b": lambda: normalized_contrast(
                local_rule_centers()[2], (local_rule_centers()[1], local_rule_centers()[3]),
            ),
            "rule_response_local_contrast_c": lambda: normalized_contrast(
                local_rule_centers()[3], local_rule_centers()[1:3],
            ),
            "rule_mass_row_permuted": lambda: (
                np.concatenate((
                    self._vector(mz, weighted, precursor, ("fragment", "neutral_loss")).astype(np.float32),
                    np.roll(
                        self._rule_response(mz, weighted, precursor).astype(np.float32),
                        1 + row % max(1, len(self.nl_rules) + len(self.cf_rules) - 1),
                    ),
                )) / np.sqrt(2.0)
            ).astype(np.float16),
        }
        result = {variant: builders[variant]() for variant in self.variants}
        self.cache[row] = result
        return result


def score_queries_truthblind(
    queries: np.ndarray, body: dict[str, np.ndarray], official: np.ndarray,
    row_position: dict[int, int], cache: KernelCache, variants: tuple[str, ...],
) -> dict:
    global_all = np.empty(len(queries), dtype=object)
    reference_ptr_all = np.empty(len(queries), dtype=object)
    kernel_all = {variant: np.empty(len(queries), dtype=object) for variant in variants}
    for out_index, query in enumerate(map(int, queries)):
        left, right = map(int, body["query_ptr"][query:query + 2])
        ref_left = int(body["molecule_ptr"][left]); ref_right = int(body["molecule_ptr"][right])
        reference_rows = body["pair_candidate_row"][ref_left:ref_right]
        local_reference_ptr = (
            body["molecule_ptr"][left:right + 1] - ref_left
        ).astype(np.int32)
        qrow = int(body["query_row"][query]); qpos = row_position[qrow]
        global_ref = official[[row_position[int(row)] for row in reference_rows]] @ official[qpos]
        qkernel = cache.get(qrow)
        ref_kernel = {
            variant: np.stack([cache.get(int(row))[variant] for row in reference_rows]).astype(np.float32)
            for variant in variants
        }
        pair_kernel = {
            variant: ref_kernel[variant] @ qkernel[variant].astype(np.float32)
            for variant in variants
        }
        global_all[out_index] = global_ref.astype(np.float32)
        reference_ptr_all[out_index] = local_reference_ptr
        for variant in variants:
            kernel_all[variant][out_index] = pair_kernel[variant].astype(np.float32)
        if (out_index + 1) % 128 == 0:
            print(f"scored {out_index + 1}/{len(queries)}", flush=True)
    return {
        "query": np.asarray(queries), "global": global_all,
        "reference_ptr": reference_ptr_all, **kernel_all,
    }


def score_queries(
    queries: np.ndarray, body: dict[str, np.ndarray], official: np.ndarray,
    row_position: dict[int, int], cache: KernelCache, variants: tuple[str, ...],
) -> dict:
    """Attach truth only after the deployment-safe score channels are frozen."""
    scored = score_queries_truthblind(queries, body, official, row_position, cache, variants)
    old_rank = np.empty(len(queries), dtype=np.int16)
    labels_all = np.empty(len(queries), dtype=object)
    for out_index, query in enumerate(map(int, queries)):
        left, right = map(int, body["query_ptr"][query:query + 2])
        labels = body["molecule_label"][left:right].astype(bool)
        pointer = np.asarray(scored["reference_ptr"][out_index], dtype=np.int64)
        molecule_global = np.maximum.reduceat(
            np.asarray(scored["global"][out_index], dtype=np.float32), pointer[:-1],
        )
        old_rank[out_index] = strict_rank(molecule_global, labels)
        labels_all[out_index] = labels
    return {
        **scored,
        "formula": body["query_formula"][queries],
        "old_rank": old_rank,
        "labels": labels_all,
    }


def fused_ranks(scored: dict, variant: str, beta: float) -> np.ndarray:
    out = np.empty(len(scored["query"]), dtype=np.int16)
    for index in range(len(out)):
        pair_score = (
            np.asarray(scored["global"][index])
            + beta * np.asarray(scored[variant][index])
        )
        reference_ptr = np.asarray(scored["reference_ptr"][index], dtype=np.int64)
        molecule_score = np.asarray([
            np.max(pair_score[left:right])
            for left, right in zip(reference_ptr[:-1], reference_ptr[1:])
        ])
        out[index] = strict_rank(
            molecule_score, np.asarray(scored["labels"][index], dtype=bool),
        )
    return out


def select_beta(scored: dict, variant: str, grid: list[float]) -> tuple[float, list[dict]]:
    table = []
    for beta in grid:
        item = metric(scored["old_rank"], fused_ranks(scored, variant, beta))
        item["beta"] = float(beta); item["risk_utility"] = item["corrected"] - 2 * item["introduced"]
        table.append(item)
    selected = max(table, key=lambda x: (x["risk_utility"], x["delta_recall1"], x["delta_mrr"], -x["beta"]))
    return float(selected["beta"]), table


def load_global_base(
    official: np.ndarray, checkpoint_path: Path | None,
) -> tuple[np.ndarray, dict]:
    if checkpoint_path is None:
        return official, {"kind": "official_dreams", "checkpoint": None}
    if torch is None:
        raise RuntimeError("Torch is required only when --adapter-checkpoint is used")
    # Optional audit-only dependency.  Direct shared-encoder training imports
    # KernelCache from this module but never uses a residual adapter.
    from dreams.models.chem_aware.global_embedding_adapter import (  # noqa: PLC0415
        GlobalResidualEmbeddingAdapter,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "chemaware_global_embedding_adapter_v1":
        raise RuntimeError("unsupported adapter checkpoint format")
    if not payload.get("validation_pass", False):
        raise RuntimeError("refusing an adapter that did not pass validation")
    config = payload["adapter_config"]
    model = GlobalResidualEmbeddingAdapter(
        int(config["dimension"]), int(config["hidden_dim"]), float(config["dropout"]),
    )
    model.load_state_dict(payload["adapter_state"], strict=True)
    model.eval()
    adapted = np.empty_like(official, dtype=np.float32)
    with torch.no_grad():
        for left in range(0, len(official), 4096):
            value = torch.from_numpy(np.array(official[left:left + 4096], copy=True))
            adapted[left:left + 4096] = model(value).cpu().numpy()
    return adapted, {
        "kind": "passed_shared_residual_adapter",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def main() -> None:
    args = arguments(); started = time.time()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    with np.load(args.manifest) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    global_base, global_base_provenance = load_global_base(official, args.adapter_checkpoint)
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    train = np.flatnonzero((fold != args.inner_fold) & (fold != args.outer_fold))
    inner_pool = np.flatnonzero(fold == args.inner_fold); outer = np.flatnonzero(fold == args.outer_fold)
    train_error, _ = official_outcomes(body, train, official, row_position, None)
    rng = np.random.default_rng(args.seed + 71)
    balanced_discovery = np.concatenate((
        identity_balanced_queries(train[train_error[train]], body["query_ik14"], rng, args.discovery_errors),
        identity_balanced_queries(train[~train_error[train]], body["query_ik14"], rng, args.discovery_correct),
    )); rng.shuffle(balanced_discovery)
    discovery = identity_balanced_queries(
        train, body["query_ik14"], np.random.default_rng(args.seed + 72),
        args.discovery_natural_identities,
    )
    inner = identity_balanced_queries(inner_pool, body["query_ik14"], np.random.default_rng(args.seed + 19), args.max_inner_identities)
    variants = tuple(dict.fromkeys(args.variants))
    controls = tuple(dict.fromkeys(args.control_variants))
    required_controls = (
        controls if args.required_control_variants is None
        else tuple(dict.fromkeys(args.required_control_variants))
    )
    if args.primary_variant not in variants:
        raise ValueError("primary variant must be included in --variants")
    if any(control not in variants for control in controls):
        raise ValueError("every control variant must be included in --variants")
    if args.primary_variant in controls:
        raise ValueError("primary variant cannot also be a control")
    if any(control not in controls for control in required_controls):
        raise ValueError("required controls must be included in control variants")
    cache = KernelCache(args, row_position, variants=variants)
    print(
        f"natural_discovery={len(discovery)} balanced_headroom={len(balanced_discovery)} "
        f"inner={len(inner)} outer_untouched={len(outer)}", flush=True,
    )
    discovery_scored = score_queries(discovery, body, global_base, row_position, cache, variants)
    inner_scored = score_queries(inner, body, global_base, row_position, cache, variants)
    selections = {}; held = {}; ranks = {}
    for variant in variants:
        beta, table = select_beta(discovery_scored, variant, list(map(float, args.fusion_grid)))
        rank = fused_ranks(inner_scored, variant, beta)
        selections[variant] = {"selected_beta": beta, "grid": table}
        ranks[variant] = rank; held[variant] = metric(inner_scored["old_rank"], rank)
    primary = args.primary_variant
    primary_delta = (ranks[primary] == 1).astype(float) - (inner_scored["old_rank"] == 1).astype(float)
    absolute_ci = formula_bootstrap(primary_delta, inner_scored["formula"], args.seed + 901, args.bootstrap_draws)
    comparisons = {}
    for offset, control in enumerate(controls):
        delta = (ranks[primary] == 1).astype(float) - (ranks[control] == 1).astype(float)
        comparisons[f"{primary}_minus_{control}"] = {
            "recall1_advantage": float(np.mean(delta)),
            **formula_bootstrap(delta, inner_scored["formula"], args.seed + 1201 + 41 * offset, args.bootstrap_draws),
        }
    gates = {
        f"{primary}_absolute_formula_ci_positive": absolute_ci["formula_cluster_bootstrap_95ci"][0] > 0,
        f"{primary}_risk_positive": held[primary]["corrected"] > held[primary]["introduced"],
    }
    gates.update({
        f"{primary}_beats_{control}": comparisons[f"{primary}_minus_{control}"]["formula_cluster_bootstrap_95ci"][0] > 0
        for control in required_controls
    })
    report = {
        "status": "DEVELOPMENT_KERNEL_PASS" if all(gates.values()) else "DEVELOPMENT_KERNEL_FAIL",
        "scope": "deployable shared-embedding kernel audit; no DreaMS weights updated; outer untouched",
        "protocol": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data": {"natural_discovery_queries": int(len(discovery)),
                 "balanced_headroom_queries_not_used_for_selection": int(len(balanced_discovery)),
                 "natural_discovery_baseline_error_fraction": float(np.mean(train_error[discovery])),
                 "inner_queries": int(len(inner)),
                 "outer_queries_untouched": int(len(outer)), "cached_rows_touched": int(len(cache.cache))},
        "shared_embedding": {"dimension": int(
                                 official.shape[1]
                                 + (
                                     args.kernel_dim * len(args.multi_bin_widths)
                                     if primary.startswith("multiscale_") else
                                     2 * args.kernel_dim if primary.startswith("mass_uniform") else
                                     args.kernel_dim + len(cache.nl_rules) + len(cache.cf_rules)
                                     if primary.startswith("rule_mass") else
                                     len(cache.nl_rules) + len(cache.cf_rules)
                                     if primary == "rule_response" else args.kernel_dim
                                 )
                             ),
                             "global_base": global_base_provenance,
                             "score_equivalence": "global_base_dot + beta * kernel_dot; common positive denominator omitted"},
        "selection_on_training_formulas": selections, "held_inner": held,
        "primary_variant": primary,
        "required_control_variants": list(required_controls),
        "primary_absolute_formula_bootstrap": absolute_ci,
        "paired_formula_cluster_comparisons": comparisons, "gates": gates,
        "contracts": {"one_spectrum_to_one_embedding": True, "candidate_set_normalization": False,
                      "candidate_structure_used": False, "same_query_reference_function": True,
                      "pair_fusion_before_molecule_max": True,
                      "formula_disjoint": True, "outer_fold_evaluated": False, "training_was_run": False},
        "provenance": {"manifest_sha256": sha256_file(args.manifest),
                       "token_report_sha256": sha256_file(args.token_dir / "report.json")},
        "runtime_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output / "inner_per_query.npz", query=inner_scored["query"], formula=inner_scored["formula"],
        old_rank=inner_scored["old_rank"], **{f"{variant}_rank": ranks[variant] for variant in variants},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
