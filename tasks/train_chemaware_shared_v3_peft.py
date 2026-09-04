"""G1-PEFT: clean full-candidate listwise fine-tuning of shared DreaMS layers.

This is the capacity-boundary follow-up to ChemAware shared embedding v2.  It
keeps the same frozen graph, formula outer/inner split, complete split-eligible
candidate groups, molecule-level max aggregation, risk penalty, and official
embedding preservation.  The only changed factor is model capacity: zero-init
low-rank updates are installed in the final DreaMS Transformer block(s) and
official projection head.  There are no chemical rules, structures, molecule
teachers, candidate features, or P2b inputs in this G1 control.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))

from chemaware_shared_v2_core import (  # noqa: E402
    ChemAwareTokenStore, MoleculeTeacherStore, formula_folds,
    identity_reference_centroids, paired_evaluation, split_allowed_molecules,
)
from dreams.models.chem_aware.frozen_probe_v3 import (  # noqa: E402
    FrozenChemicalProbe, fit_frozen_ridge_probe, targeted_probe_listwise_loss,
    targeted_probe_multiview_listwise_loss,
)
from dreams.models.chem_aware.peft_v3 import (  # noqa: E402
    DreaMSPEFTConfig, install_dreams_peft, load_peft_state_dict, peft_state_dict,
)
from dreams.models.chem_aware.peak_rule_attention_v3 import (  # noqa: E402
    PeakRuleBiasStore,
)
from dreams.models.chem_aware.shared_embedding_v2 import (  # noqa: E402
    chemical_margin_listwise_loss, chemical_weighted_listwise_loss,
    molecule_listwise_loss, molecule_listwise_loss_per_query,
    molecule_scores_from_spectrum_pairs, protected_margin_loss,
    positive_reference_increment_loss, targeted_chemical_margin_increment,
)
from noise_final_core import CandidateGraph, json_dump, seed_everything, sha256_file  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402
from train_chemaware_shared_v2 import (  # noqa: E402
    build_query_batch, gradient_geometry, state_sha256,
)


FROZEN_PROBE_OBJECTIVES = {
    "frozen_probe_targeted",
    "frozen_probe_targeted_multiview",
}


def model_selection_eligible(
    summary: dict,
    *,
    minimum_preservation: float,
    minimum_single_spectrum_preservation: float,
    recall1_floor: float,
    near_recall1_floor: float,
) -> bool:
    """Apply every safety floor used to admit an epoch for model selection.

    Keeping this gate pure and separately testable prevents a high mean
    preservation score from silently masking one badly drifted spectrum.
    """

    near_delta = summary["delta_near_recall1"]
    return bool(
        summary["preservation_mean"] >= minimum_preservation
        and summary["preservation_min"] >= minimum_single_spectrum_preservation
        and summary["delta_recall1"] >= recall1_floor
        and (near_delta is None or near_delta >= near_recall1_floor)
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_tokens")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--raw-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument(
        "--initial-peft-checkpoint", type=Path,
        help=(
            "local staged diagnostic: warm-start from a clean G1 PEFT checkpoint; "
            "epoch 0 remains the selected fallback unless validation utility improves"
        ),
    )
    parser.add_argument("--preflight", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_preflight.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v3_peft_g1")
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1701)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-queries", type=int, default=4)
    parser.add_argument("--forward-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--last-blocks", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument(
        "--adapt-attention", action=argparse.BooleanOptionalAction, default=True,
        help="install low-rank updates on attention projection matrices",
    )
    parser.add_argument(
        "--adapt-feed-forward", action=argparse.BooleanOptionalAction, default=True,
        help="install low-rank updates on Transformer feed-forward matrices",
    )
    parser.add_argument(
        "--adapt-head", action=argparse.BooleanOptionalAction, default=True,
        help="install a low-rank update on the shared projection head",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-listwise", type=float, default=1.0)
    parser.add_argument("--lambda-protect", type=float, default=2.0)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument(
        "--lambda-positive-increment", type=float, default=0.0,
        help=(
            "weight of a training-only same-identity cross-spectrum similarity "
            "increment anchored to frozen official DreaMS geometry"
        ),
    )
    parser.add_argument(
        "--positive-increment", type=float, default=0.01,
        help="requested cosine increment over each official positive-reference pair",
    )
    parser.add_argument(
        "--positive-increment-aggregation", choices=("mean", "worst"),
        default="mean",
        help="aggregate all positive views or emphasize the least-improved view",
    )
    parser.add_argument(
        "--lambda-worst-preserve", type=float, default=0.0,
        help=(
            "weight of a batchwise worst-spectrum cosine-drift hinge; zero "
            "reproduces the original mean-only preservation objective"
        ),
    )
    parser.add_argument(
        "--worst-preserve-slack", type=float, default=0.03,
        help="allowed worst-spectrum cosine drift before the hinge becomes active",
    )
    parser.add_argument("--protect-slack", type=float, default=0.0)
    parser.add_argument("--minimum-preservation", type=float, default=0.995)
    parser.add_argument(
        "--minimum-single-spectrum-preservation", type=float, default=0.95,
        help="fail model selection when any graph-reachable spectrum drifts below this cosine",
    )
    parser.add_argument("--recall1-floor", type=float, default=-5e-4)
    parser.add_argument("--near-recall1-floor", type=float, default=-1e-3)
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--near-selection-weight", type=float, default=0.25)
    parser.add_argument(
        "--margin-selection-weight", type=float, default=0.0,
        help=(
            "inner-only continuous tie-breaker weight for the mean positive-vs-best-"
            "negative cosine margin; zero reproduces the original rank-only selector"
        ),
    )
    parser.add_argument(
        "--official-error-focus-strength", type=float, default=0.0,
        help=(
            "bounded extra training weight for frozen-official low-margin/error "
            "queries; zero is the original identity-equal G1 curriculum"
        ),
    )
    parser.add_argument(
        "--official-error-focus-temperature", type=float, default=0.02,
        help="cosine-margin scale for the deterministic official-error curriculum",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--molecule-teacher-dir", type=Path)
    parser.add_argument(
        "--teacher-control",
        choices=(
            "correct", "identity_permuted", "random_marginal",
            "correct_same_formula_scope", "same_formula_mismatched",
        ),
    )
    parser.add_argument("--lambda-molecule", type=float, default=0.25)
    parser.add_argument("--molecule-temperature", type=float, default=0.07)
    parser.add_argument("--chemical-hardness-beta", type=float, default=4.0)
    parser.add_argument(
        "--chemical-weighting",
        choices=("relative_centered", "absolute_bounded"),
        default="absolute_bounded",
    )
    parser.add_argument(
        "--chemical-objective",
        choices=(
            "candidate_hardness", "candidate_margin",
            "candidate_margin_targeted",
            *sorted(FROZEN_PROBE_OBJECTIVES),
        ),
        default="candidate_hardness",
    )
    parser.add_argument(
        "--chemical-margin-scale", type=float, default=0.03,
        help="maximum raw-cosine additive negative margin for candidate-margin diagnostic",
    )
    parser.add_argument(
        "--chemical-margin-similarity-threshold", type=float, default=0.40,
        help="teacher cosine below which targeted candidate margin is exactly zero",
    )
    parser.add_argument("--lambda-probe", type=float, default=0.05)
    parser.add_argument("--probe-alpha", type=float, default=1.0)
    parser.add_argument("--probe-temperature", type=float, default=0.1)
    parser.add_argument("--probe-margin-threshold", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--frozen-prefix-cache", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "local diagnostic accelerator: cache the frozen prefix before the only "
            "adapted Transformer block and train/evaluate just that block plus head"
        ),
    )
    parser.add_argument(
        "--prefix-cache-batch-size", type=int, default=8,
        help="batch size used once to materialize the frozen-prefix cache",
    )
    parser.add_argument(
        "--peak-rule-view-scale", type=float, default=0.0,
        help=(
            "local G2f diagnostic: fixed additive chemical attention-logit "
            "bias used only in an auxiliary training view; zero disables it"
        ),
    )
    parser.add_argument(
        "--peak-rule-bias-kind",
        choices=("binary_union", "idf_precursor"),
        default="binary_union",
        help=(
            "legacy all-edge binary union or IDF-weighted precursor-to-fragment "
            "NL/CF prior"
        ),
    )
    parser.add_argument(
        "--peak-rule-categories",
        default="NL,CF,ISO",
        help="comma-separated NL/CF/ISO subset; idf_precursor supports NL/CF",
    )
    parser.add_argument(
        "--peak-rule-control",
        choices=("correct", "spectrum_permuted", "peak_permuted"),
        help=(
            "correct alignment, whole-spectrum permutation, or exact within-spectrum "
            "fragment-token permutation attribution control"
        ),
    )
    parser.add_argument(
        "--lambda-peak-rule", type=float, default=0.25,
        help="convex weight of the training-only peak-rule retrieval view",
    )
    parser.add_argument(
        "--peak-rule-objective",
        choices=(
            "auxiliary_listwise", "matched_preference",
            "matched_evidence_preference",
        ),
        default="auxiliary_listwise",
        help=(
            "train on the chemical view directly, or require the correctly aligned "
            "view to beat an exact within-spectrum peak-permuted control"
        ),
    )
    parser.add_argument(
        "--peak-rule-preference-margin", type=float, default=0.01,
        help="listwise-loss gap required by the matched peak-rule preference hinge",
    )
    parser.add_argument(
        "--peak-rule-min-evidence", type=float, default=0.0,
        help=(
            "for matched_evidence_preference, activate only queries whose detached "
            "control-minus-correct loss exceeds this threshold"
        ),
    )
    parser.add_argument("--max-train-queries", type=int, default=0, help="non-formal smoke only")
    parser.add_argument("--max-eval-queries", type=int, default=0, help="non-formal smoke only")
    return parser.parse_args()


class RawSpectrumStore:
    """Preprocess every graph-reachable spectrum once on CPU."""

    def __init__(self, data: Path, rows: np.ndarray, n_highest_peaks: int):
        self.rows = np.asarray(rows, dtype=np.int64)
        if self.rows.ndim != 1 or len(np.unique(self.rows)) != len(self.rows):
            raise RuntimeError("raw-spectrum rows must be a unique vector")
        self.position = {int(row): index for index, row in enumerate(self.rows)}
        tensors = []
        with h5py.File(data, "r") as handle:
            for row in self.rows:
                tensors.append(preprocess_spectrum(
                    np.asarray(handle["spectrum"][int(row)]),
                    float(handle["precursor_mz"][int(row)]),
                    n_highest_peaks,
                ))
        self.tensor = torch.stack(tensors)

    def get(self, rows: np.ndarray) -> torch.Tensor:
        try:
            positions = [self.position[int(row)] for row in np.asarray(rows, dtype=np.int64)]
        except KeyError as error:
            raise RuntimeError(f"spectrum row absent from raw cache: {error}") from error
        return self.tensor[positions]


class FrozenPrefixSpectrumStore:
    """Exact shared-encoder accelerator for adapted final DreaMS blocks.

    The frozen prefix layers are executed once.  Their output and the frozen
    Graphormer bias are cached on CPU; every subsequent call still runs all
    deployable adapted final attention/FFN blocks and the projection head.
    This changes computation only, not the model or inference contract.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        source: RawSpectrumStore,
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

        self.rows = source.rows.copy()
        self.position = dict(source.position)
        prefix_chunks = []
        bias_chunks = []
        mask_chunks = []
        model.eval()
        dtype = next(model.parameters()).dtype
        captured: list[torch.Tensor] = []

        def capture_prefix(_module, inputs):
            captured.append(inputs[0].detach())

        handle = encoder.scales[2 * start_layer].register_forward_pre_hook(capture_prefix)
        try:
            with torch.no_grad():
                for left in range(0, len(self.rows), batch_size):
                    rows = self.rows[left:left + batch_size]
                    spectra = source.get(rows).to(device=device, dtype=dtype)
                    captured.clear()
                    model.backbone(spectra, None)
                    if len(captured) != 1:
                        raise RuntimeError("failed to capture exactly one final-layer prefix")
                    prefix_chunks.append(captured[0].float().cpu())
                    mask_chunks.append((spectra[:, :, 0] == 0).cpu())
                    # Linear(graph_i - graph_j) is exactly the intended frozen
                    # Graphormer bias up to ordinary floating-point roundoff,
                    # without materializing the 980-dimensional pair tensor.
                    fourier = backbone.ff_fourier(
                        backbone.fourier_enc(spectra[..., [0]])
                    )
                    projected = torch.sum(fourier, dim=-1, keepdim=True)
                    bias = projected.unsqueeze(2) - projected.unsqueeze(1)
                    # Store as a one-channel ``graphormer_dists`` tensor. The
                    # official attention sums its last dimension; a chemical
                    # view, when enabled, is added separately through the now
                    # audited ``chem_bias`` attention-logit path.
                    bias_chunks.append(bias.float().cpu())
        finally:
            handle.remove()
        self.prefix = torch.cat(prefix_chunks, dim=0).contiguous()
        self.graphormer_bias = torch.cat(bias_chunks, dim=0).contiguous()
        self.padding_mask = torch.cat(mask_chunks, dim=0).contiguous()
        self.start_layer = start_layer
        self.final_layer = final_layer
        self.adapted_layers = adapted_layers
        self.audit = {
            "enabled": True,
            "cached_spectra": int(len(self.rows)),
            "start_layer": start_layer,
            "final_layer": final_layer,
            "adapted_layers": list(adapted_layers),
            "prefix_shape": list(self.prefix.shape),
            "graphormer_bias_shape": list(self.graphormer_bias.shape),
            "prefix_bytes": int(self.prefix.numel() * self.prefix.element_size()),
            "graphormer_bias_bytes": int(
                self.graphormer_bias.numel() * self.graphormer_bias.element_size()
            ),
            "training_only_computation_cache": True,
        }

    def forward(
        self,
        model: torch.nn.Module,
        rows: np.ndarray,
        device: torch.device,
        batch_size: int,
        amp: bool,
        peak_rule_store: PeakRuleBiasStore | None = None,
    ) -> torch.Tensor:
        try:
            positions = np.asarray(
                [self.position[int(row)] for row in np.asarray(rows, dtype=np.int64)],
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
            bias = self.graphormer_bias[index].to(device=device, dtype=dtype)
            peak_rule_bias = (
                None
                if peak_rule_store is None
                else peak_rule_store.get(
                    self.rows[index], device=device, dtype=dtype
                )
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=bool(amp and device.type == "cuda"),
            ):
                for layer in self.adapted_layers:
                    x = encoder._layer_forward(
                        layer, x, mask, bias,
                        chem_bias=peak_rule_bias,
                    )
                x = encoder.scales[-1](x)
                outputs.append(torch.nn.functional.normalize(model.head(x[:, 0]), dim=-1))
        return torch.cat(outputs, dim=0)


def forward_rows(
    model: torch.nn.Module,
    raw_store: RawSpectrumStore,
    rows: np.ndarray,
    device: torch.device,
    batch_size: int,
    amp: bool,
    peak_rule_store: PeakRuleBiasStore | None = None,
) -> torch.Tensor:
    """Encode unique rows in microbatches without detaching PEFT gradients."""

    if isinstance(raw_store, FrozenPrefixSpectrumStore):
        return raw_store.forward(
            model, rows, device, batch_size, amp,
            peak_rule_store=peak_rule_store,
        )

    if peak_rule_store is not None:
        raise RuntimeError(
            "peak-rule attention view currently requires the audited frozen-prefix path"
        )

    outputs = []
    dtype = next(model.parameters()).dtype
    for left in range(0, len(rows), batch_size):
        spectra = raw_store.get(rows[left:left + batch_size]).to(
            device=device, dtype=dtype, non_blocking=True
        )
        with torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=bool(amp and device.type == "cuda"),
        ):
            outputs.append(model(spectra))
    return torch.cat(outputs, dim=0)


def score_batch(
    model: torch.nn.Module,
    raw_store: RawSpectrumStore,
    official_store: ChemAwareTokenStore,
    batch: dict[str, np.ndarray],
    device: torch.device,
    forward_batch_size: int,
    amp: bool,
    peak_rule_store: PeakRuleBiasStore | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]:
    joined = np.concatenate((batch["query_rows"], batch["candidate_rows"]))
    unique, inverse = np.unique(joined, return_inverse=True)
    encoded = forward_rows(
        model, raw_store, unique, device, forward_batch_size, amp,
        peak_rule_store=peak_rule_store,
    ).float()
    official = official_store.tensors(unique, device)[0]
    n_query = len(batch["query_rows"])
    query_index = torch.from_numpy(inverse[:n_query]).to(device=device, dtype=torch.long)
    candidate_index = torch.from_numpy(inverse[n_query:]).to(device=device, dtype=torch.long)
    pair_query = torch.from_numpy(batch["pair_query"]).to(device=device, dtype=torch.long)
    molecule_ptr = torch.from_numpy(batch["molecule_ptr"]).to(device=device, dtype=torch.long)
    new_scores = molecule_scores_from_spectrum_pairs(
        encoded[query_index], encoded[candidate_index], pair_query, molecule_ptr
    )
    old_scores = molecule_scores_from_spectrum_pairs(
        official[query_index], official[candidate_index], pair_query, molecule_ptr
    )
    # Roundoff can put a cosine a few ulps above one.  A preservation penalty
    # must never reward that numerical accident with a negative loss.
    spectrum_drift = torch.clamp(
        1.0 - torch.sum(encoded * official, dim=1), min=0.0
    )
    preservation = torch.mean(spectrum_drift)
    worst_preservation = torch.max(spectrum_drift)
    positive_reference_parts = []
    official_positive_reference_parts = []
    positive_reference_ptr = [0]
    for query_index_local in range(n_query):
        positive_molecule = int(batch["query_ptr"][query_index_local])
        pair_left = int(batch["molecule_ptr"][positive_molecule])
        pair_right = int(batch["molecule_ptr"][positive_molecule + 1])
        positive_reference_parts.append(encoded[candidate_index[pair_left:pair_right]])
        official_positive_reference_parts.append(
            official[candidate_index[pair_left:pair_right]]
        )
        positive_reference_ptr.append(
            positive_reference_ptr[-1] + pair_right - pair_left
        )
    positive_reference_embeddings = torch.cat(positive_reference_parts, dim=0)
    official_positive_reference_embeddings = torch.cat(
        official_positive_reference_parts, dim=0
    )
    positive_reference_ptr_tensor = torch.tensor(
        positive_reference_ptr, device=device, dtype=torch.long
    )
    return (
        new_scores, old_scores, preservation, encoded[query_index],
        positive_reference_embeddings, positive_reference_ptr_tensor,
        official[query_index], official_positive_reference_embeddings,
        worst_preservation,
    )


@torch.no_grad()
def encode_all(
    model: torch.nn.Module,
    raw_store: RawSpectrumStore,
    device: torch.device,
    batch_size: int,
    amp: bool,
    peak_rule_store: PeakRuleBiasStore | None = None,
) -> np.ndarray:
    model.eval()
    values = forward_rows(
        model, raw_store, raw_store.rows, device, batch_size, amp,
        peak_rule_store=peak_rule_store,
    )
    result = values.float().cpu().numpy()
    if not np.all(np.isfinite(result)):
        raise RuntimeError("PEFT encoder produced non-finite embeddings")
    norm_error = float(np.max(np.abs(np.linalg.norm(result, axis=1) - 1.0)))
    if norm_error > 2e-3:
        raise RuntimeError(f"PEFT embeddings are not normalized: max error={norm_error}")
    return result


def _validate_args(args: argparse.Namespace) -> None:
    if args.folds < 3 or args.outer_fold not in range(args.folds):
        raise ValueError("invalid formula outer fold")
    positive = (
        args.epochs, args.batch_queries, args.forward_batch_size,
        args.eval_batch_size, args.n_highest_peaks, args.last_blocks,
        args.rank, args.alpha, args.lr, args.temperature, args.grad_clip,
        args.prefix_cache_batch_size,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("training sizes, PEFT capacity, learning rate, and temperature must be positive")
    if not (args.adapt_attention or args.adapt_feed_forward or args.adapt_head):
        raise ValueError("at least one PEFT target must be enabled")
    if (args.molecule_teacher_dir is None) != (args.teacher_control is None):
        raise ValueError("molecule-teacher-dir and teacher-control must be supplied together")
    peak_rule_enabled = args.peak_rule_view_scale > 0
    args.peak_rule_categories = tuple(
        item.strip().upper()
        for item in str(args.peak_rule_categories).split(",")
        if item.strip()
    )
    if (
        not args.peak_rule_categories
        or not set(args.peak_rule_categories) <= {"NL", "CF", "ISO"}
    ):
        raise ValueError("peak-rule-categories must be a nonempty NL/CF/ISO subset")
    if (
        args.peak_rule_bias_kind == "idf_precursor"
        and not set(args.peak_rule_categories) <= {"NL", "CF"}
    ):
        raise ValueError("idf_precursor supports NL and CF categories only")
    if args.peak_rule_view_scale < 0:
        raise ValueError("peak-rule-view-scale must be nonnegative")
    if peak_rule_enabled != (args.peak_rule_control is not None):
        raise ValueError(
            "nonzero peak-rule-view-scale and peak-rule-control must be supplied together"
        )
    if peak_rule_enabled and args.lambda_peak_rule <= 0:
        raise ValueError("lambda-peak-rule must be positive")
    if (
        peak_rule_enabled
        and args.peak_rule_objective == "auxiliary_listwise"
        and args.lambda_peak_rule > 1
    ):
        raise ValueError("auxiliary-listwise lambda-peak-rule must be in (0, 1]")
    if args.peak_rule_preference_margin < 0:
        raise ValueError("peak-rule-preference-margin must be nonnegative")
    if args.peak_rule_min_evidence < 0:
        raise ValueError("peak-rule-min-evidence must be nonnegative")
    if (
        peak_rule_enabled
        and args.peak_rule_objective.startswith("matched_")
        and args.peak_rule_control != "correct"
    ):
        raise ValueError("matched peak-rule preference requires the correct primary view")
    if peak_rule_enabled and args.molecule_teacher_dir is not None:
        raise ValueError("peak-rule view and molecule teacher must be tested in separate arms")
    if peak_rule_enabled and not args.frozen_prefix_cache:
        raise ValueError("peak-rule view currently requires --frozen-prefix-cache")
    if args.molecule_teacher_dir is not None and not (0 < args.lambda_molecule <= 1):
        raise ValueError("G2 molecule convex weight must be in (0, 1]")
    if args.chemical_hardness_beta < 0:
        raise ValueError("chemical-hardness-beta must be nonnegative")
    if args.chemical_margin_scale < 0:
        raise ValueError("chemical-margin-scale must be nonnegative")
    if not 0 <= args.chemical_margin_similarity_threshold < 1:
        raise ValueError("chemical margin similarity threshold must be in [0, 1)")
    if (
        args.official_error_focus_strength < 0
        or args.official_error_focus_temperature <= 0
    ):
        raise ValueError("invalid official-error focus curriculum")
    if args.lambda_worst_preserve < 0 or not 0 <= args.worst_preserve_slack < 1:
        raise ValueError("invalid worst-spectrum preservation contract")
    if args.margin_selection_weight < 0:
        raise ValueError("margin-selection-weight must be nonnegative")
    if args.lambda_positive_increment < 0 or args.positive_increment < 0:
        raise ValueError("invalid positive-reference increment contract")
    if args.molecule_teacher_dir is None and args.chemical_objective != "candidate_hardness":
        raise ValueError("a chemical objective requires molecule-teacher-dir")
    if args.chemical_objective in FROZEN_PROBE_OBJECTIVES and (
        args.lambda_probe <= 0 or args.probe_alpha <= 0
        or args.probe_temperature <= 0 or args.probe_margin_threshold < 0
    ):
        raise ValueError("invalid frozen-probe chemical contract")
    if not (
        0 < args.minimum_single_spectrum_preservation
        <= args.minimum_preservation <= 1
    ):
        raise ValueError("invalid mean/single-spectrum preservation contract")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("ChemAware shared-v3 PEFT training requires an available CUDA device")
    if args.frozen_prefix_cache and args.last_blocks < 1:
        raise ValueError("frozen-prefix cache requires at least one adapted final block")
    if args.peak_rule_view_scale > 0 and args.last_blocks != 1:
        raise ValueError(
            "peak-rule attention remains restricted to one adapted block until "
            "multi-block chemical-view attribution is separately audited"
        )


def targeted_probe_query_audit(
    graph: CandidateGraph,
    train_query: np.ndarray,
    allowed_molecule: np.ndarray,
    teacher_observable: np.ndarray,
    margin_threshold: float,
) -> dict:
    """Audit the label-free official-margin selection used by every probe arm."""

    active = np.zeros(graph.n_queries, dtype=bool)
    official_wrong = low_margin_correct = 0
    margins = []
    for query in np.asarray(train_query, dtype=np.int64):
        left, right = map(int, graph.query_ptr[query:query + 2])
        selected = np.flatnonzero(allowed_molecule[left:right]) + left
        scores = np.asarray([
            np.max(graph.features[
                int(graph.molecule_ptr[molecule]):int(graph.molecule_ptr[molecule + 1]),
                graph.dreams_column,
            ])
            for molecule in selected
        ], dtype=np.float32)
        margin = float(scores[0] - np.max(scores[1:]))
        margins.append(margin)
        observable = teacher_observable[selected]
        usable = bool(observable[0] and np.any(observable[1:]))
        if usable and margin <= margin_threshold:
            active[query] = True
            if margin <= 0:
                official_wrong += 1
            else:
                low_margin_correct += 1
    active_queries = np.flatnonzero(active)
    return {
        "active_mask": active,
        "active_queries": active_queries,
        "chemical_effect_queries": int(len(active_queries)),
        "chemical_effect_query_fraction": float(len(active_queries) / len(train_query)),
        "official_wrong_queries": int(official_wrong),
        "low_margin_official_correct_queries": int(low_margin_correct),
        "high_margin_official_correct_queries_selected": 0,
        "margin_threshold": float(margin_threshold),
        "official_margin_min": float(np.min(margins)),
        "official_margin_median": float(np.median(margins)),
        "selection_query_ledger_sha256": hashlib.sha256(
            np.ascontiguousarray(active_queries, dtype=np.int64).tobytes()
        ).hexdigest(),
    }


def targeted_margin_query_audit(
    graph: CandidateGraph,
    train_query: np.ndarray,
    allowed_molecule: np.ndarray,
    teacher_observable: np.ndarray,
) -> dict:
    """Frozen-official error selection, independent of teacher target values."""

    selected_queries = []
    for query in np.asarray(train_query, dtype=np.int64):
        left, right = map(int, graph.query_ptr[query:query + 2])
        selected = np.flatnonzero(allowed_molecule[left:right]) + left
        scores = np.asarray([
            np.max(graph.features[
                int(graph.molecule_ptr[molecule]):int(graph.molecule_ptr[molecule + 1]),
                graph.dreams_column,
            ])
            for molecule in selected
        ], dtype=np.float32)
        observable = teacher_observable[selected]
        if scores[0] <= np.max(scores[1:]) and observable[0] and np.any(observable[1:]):
            selected_queries.append(int(query))
    selected_queries = np.asarray(selected_queries, dtype=np.int64)
    return {
        "selected_official_error_queries": int(len(selected_queries)),
        "high_margin_official_correct_queries_selected": 0,
        "selection_query_ledger_sha256": hashlib.sha256(
            np.ascontiguousarray(selected_queries).tobytes()
        ).hexdigest(),
        "selection_uses_teacher_target_values": False,
    }


def official_error_focus_weights(
    graph: CandidateGraph,
    train_query: np.ndarray,
    allowed_molecule: np.ndarray,
    strength: float,
    temperature: float,
) -> tuple[dict[int, float], dict]:
    """Build deterministic identity-balanced weights from frozen train margins.

    This is deliberately nonchemical.  It is the matched control required
    before any structure-aware hard-negative curriculum can claim additional
    value.  The base ``1 / spectra_per_identity`` weight is retained and a
    bounded multiplier in ``[1, 1 + strength]`` focuses official errors and
    low-margin queries.  Only training-fold labels and frozen official scores
    are used.
    """

    train_query = np.asarray(train_query, dtype=np.int64)
    if strength < 0 or temperature <= 0 or not len(train_query):
        raise ValueError("invalid official-error focus inputs")
    margins = []
    for query in train_query:
        left, right = map(int, graph.query_ptr[query:query + 2])
        selected = np.flatnonzero(allowed_molecule[left:right]) + left
        if not len(selected) or selected[0] != left or len(selected) < 2:
            raise RuntimeError(f"invalid split-eligible query group: {int(query)}")
        scores = np.asarray([
            np.max(graph.features[
                int(graph.molecule_ptr[molecule]):int(graph.molecule_ptr[molecule + 1]),
                graph.dreams_column,
            ])
            for molecule in selected
        ], dtype=np.float64)
        margins.append(float(scores[0] - np.max(scores[1:])))
    margins = np.asarray(margins, dtype=np.float64)
    # Stable sigmoid(-margin / temperature): large negative margins approach 1.
    scaled = np.clip(margins / temperature, -50.0, 50.0)
    focus_multiplier = 1.0 + strength / (1.0 + np.exp(scaled))

    identities = graph.query_ik14[train_query].astype(str)
    _, identity_inverse, identity_count = np.unique(
        identities, return_inverse=True, return_counts=True
    )
    weights = focus_multiplier / identity_count[identity_inverse]
    weights /= np.mean(weights)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("official-error focus produced invalid query weights")
    ledger = np.column_stack((train_query.astype(np.float64), weights))
    wrong = margins <= 0
    weight_by_query = {
        int(query): float(weight) for query, weight in zip(train_query, weights)
    }
    return weight_by_query, {
        "kind": (
            "identity_balanced_frozen_official_error_focus"
            if strength > 0 else "identity_equal"
        ),
        "strength": float(strength),
        "temperature": float(temperature),
        "train_queries": int(len(train_query)),
        "official_wrong_queries": int(np.sum(wrong)),
        "official_correct_queries": int(np.sum(~wrong)),
        "official_margin_min": float(np.min(margins)),
        "official_margin_median": float(np.median(margins)),
        "official_margin_max": float(np.max(margins)),
        "focus_multiplier_min": float(np.min(focus_multiplier)),
        "focus_multiplier_max": float(np.max(focus_multiplier)),
        "mean_final_weight_official_wrong": (
            float(np.mean(weights[wrong])) if np.any(wrong) else None
        ),
        "mean_final_weight_official_correct": (
            float(np.mean(weights[~wrong])) if np.any(~wrong) else None
        ),
        "query_weight_ledger_sha256": hashlib.sha256(
            np.ascontiguousarray(ledger, dtype=np.float64).tobytes()
        ).hexdigest(),
        "uses_heldout_queries": False,
        "uses_structure_teacher": False,
    }


def main() -> None:
    args = arguments()
    _validate_args(args)
    required = (
        args.graph, args.data, args.official_checkpoint, args.raw_checkpoint,
        args.token_dir / "report.json",
    )
    required = list(required)
    if args.molecule_teacher_dir is not None:
        required.append(args.molecule_teacher_dir / "report.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    formal = args.max_train_queries == 0 and args.max_eval_queries == 0
    if formal and args.chemical_objective == "frozen_probe_targeted_multiview":
        raise RuntimeError(
            "multiview frozen-probe is a rejected mechanism diagnostic, not a formal objective"
        )
    if formal and args.chemical_objective in {
        "candidate_margin", "candidate_margin_targeted",
    }:
        raise RuntimeError(
            "candidate-margin is a local mechanism diagnostic until matched controls pass"
        )
    if formal and args.frozen_prefix_cache:
        raise RuntimeError(
            "frozen-prefix cache remains a local accelerator until equivalence is audited"
        )
    if formal and args.peak_rule_view_scale > 0:
        raise RuntimeError(
            "peak-rule attention is a local mechanism diagnostic until matched controls pass"
        )
    if formal and args.initial_peft_checkpoint is not None:
        raise RuntimeError(
            "warm-start PEFT remains a local staged diagnostic until paired audits pass"
        )
    if formal:
        if not args.preflight.is_file():
            raise FileNotFoundError(args.preflight)
        preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
        if (
            preflight.get("status") != "chemaware_corrected_training_preflight_passed"
            or preflight.get("formal") is not True
            or preflight.get("data_contract") != "train_primary_all_p3_disjoint_v1"
        ):
            raise RuntimeError(
                "formal v3 training requires a corrected train_primary_all P3-disjoint "
                "preflight; the historical frozen-v2 preflight cannot authorize new training"
            )
        expected = preflight.get("hashes", {})
        observed = {
            "graph_sha256": sha256_file(args.graph),
            "hdf5_sha256": sha256_file(args.data),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
        }
        if any(expected.get(key) != value for key, value in observed.items()):
            raise RuntimeError("formal v3 inputs differ from frozen preflight hashes")
        token_report = json.loads((args.token_dir / "report.json").read_text(encoding="utf-8"))
        if token_report.get("provenance", {}).get("preflight_sha256") != sha256_file(args.preflight):
            raise RuntimeError("formal v3 token cache belongs to a different preflight")
        if args.molecule_teacher_dir is not None:
            teacher_report = json.loads(
                (args.molecule_teacher_dir / "report.json").read_text(encoding="utf-8")
            )
            if teacher_report.get("provenance", {}).get("preflight_sha256") != sha256_file(args.preflight):
                raise RuntimeError("formal v3 molecule teacher belongs to a different preflight")

    experiment = (
        (
            f"G2f_peak_rule_attention_{args.peak_rule_bias_kind}_{args.peak_rule_objective}_{args.peak_rule_control}"
            if args.peak_rule_view_scale > 0
            else (
                "G1p_positive_pair_increment_listwise_PEFT_control"
                if args.lambda_positive_increment > 0
                else (
                    "G1f_official_error_focused_listwise_PEFT_control"
                    if args.official_error_focus_strength > 0
                    else "G1_clean_listwise_PEFT_capacity_control"
                )
            )
        )
        if args.molecule_teacher_dir is None
        else (
            f"G2b_PEFT_{args.chemical_objective}_{args.teacher_control}"
            if args.chemical_objective in FROZEN_PROBE_OBJECTIVES
            else (
                (
                    f"G2d_PEFT_targeted_candidate_margin_{args.teacher_control}"
                    if args.chemical_objective == "candidate_margin_targeted"
                    else f"G2c_PEFT_candidate_margin_{args.teacher_control}"
                )
                if args.chemical_objective in {"candidate_margin", "candidate_margin_targeted"}
                else f"G2_PEFT_structure_teacher_{args.teacher_control}"
            )
        )
    )
    output_base = (
        (
            args.output_root / "peak_rule_attention" / str(args.peak_rule_bias_kind) / str(args.peak_rule_objective) / str(args.peak_rule_control)
            if args.peak_rule_view_scale > 0 else args.output_root
        ) if args.molecule_teacher_dir is None
        else (
            args.output_root / args.chemical_objective / str(args.teacher_control)
            if args.chemical_objective in FROZEN_PROBE_OBJECTIVES
            else (
                args.output_root / args.chemical_objective / str(args.teacher_control)
                if args.chemical_objective in {
                    "candidate_margin", "candidate_margin_targeted",
                }
                else args.output_root / str(args.teacher_control)
            )
        )
    )
    output = output_base / f"seed_{args.seed}" / f"fold_{args.outer_fold}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite ChemAware v3 output: {output}")
    seed_everything(args.seed)
    device = torch.device(args.device)
    graph = CandidateGraph(args.graph)
    official_store = ChemAwareTokenStore(
        args.token_dir, args.graph, args.official_checkpoint, require_formal=formal
    )
    official_store.require_graph_coverage(graph)
    graph_score_error = official_store.verify_official_graph_scores(graph)
    raw_store = RawSpectrumStore(args.data, official_store.rows, args.n_highest_peaks)
    peak_rule_store = None
    peak_rule_control_store = None
    peak_rule_audit = None
    if args.peak_rule_view_scale > 0:
        peak_rule_store = PeakRuleBiasStore(
            raw_store,
            scale=args.peak_rule_view_scale,
            control=str(args.peak_rule_control),
            seed=args.seed + 1009 * args.outer_fold,
            categories=args.peak_rule_categories,
            bias_kind=args.peak_rule_bias_kind,
        )
        if args.peak_rule_objective.startswith("matched_"):
            peak_rule_control_store = PeakRuleBiasStore(
                raw_store,
                scale=args.peak_rule_view_scale,
                control="peak_permuted",
                seed=args.seed + 1009 * args.outer_fold,
                categories=args.peak_rule_categories,
                bias_kind=args.peak_rule_bias_kind,
            )
        peak_rule_audit = {
            "primary": peak_rule_store.audit,
            "matched_control": (
                peak_rule_control_store.audit
                if peak_rule_control_store is not None else None
            ),
            "objective": args.peak_rule_objective,
            "preference_margin": args.peak_rule_preference_margin,
        }

    query_fold = formula_folds(graph.query_formula, args.folds, args.fold_seed)
    inner_fold = (args.outer_fold + 1) % args.folds
    train_query = np.flatnonzero(
        (query_fold != args.outer_fold) & (query_fold != inner_fold)
    )
    inner_query = np.flatnonzero(query_fold == inner_fold)
    outer_query = np.flatnonzero(query_fold == args.outer_fold)
    allowed_molecule = split_allowed_molecules(
        graph, args.outer_fold, inner_fold, args.folds, args.fold_seed
    )
    train_query = np.asarray([
        int(query) for query in train_query
        if allowed_molecule[int(graph.query_ptr[query])]
        and int(np.sum(allowed_molecule[
            int(graph.query_ptr[query]):int(graph.query_ptr[query + 1])
        ])) >= 2
    ], dtype=np.int64)
    if args.max_train_queries:
        train_query = train_query[:args.max_train_queries]
    if args.max_eval_queries:
        inner_query = inner_query[:args.max_eval_queries]
        outer_query = outer_query[:args.max_eval_queries]
    if not len(train_query) or not len(inner_query) or not len(outer_query):
        raise RuntimeError("empty train/inner/outer formula split")

    model, initialization_kind = load_base_model(
        args.official_checkpoint, args.raw_checkpoint, device, args.n_highest_peaks
    )
    model.eval()  # Fixed dropout is part of the controlled PEFT comparison.
    peft_config = DreaMSPEFTConfig(
        last_blocks=args.last_blocks, rank=args.rank, alpha=args.alpha,
        adapt_attention=args.adapt_attention,
        adapt_feed_forward=args.adapt_feed_forward,
        adapt_head=args.adapt_head,
    )
    capacity = install_dreams_peft(model, peft_config)
    warm_start_audit = None
    if args.initial_peft_checkpoint is not None:
        if not args.initial_peft_checkpoint.is_file():
            raise FileNotFoundError(args.initial_peft_checkpoint)
        warm = torch.load(
            args.initial_peft_checkpoint, map_location="cpu", weights_only=False
        )
        required_warm = {
            "format": "chemaware_shared_v3_peft_v1",
            "chemical_supervision": False,
            "candidate_inputs_at_inference": False,
            "query_reference_encoder_shared": True,
            "P2b_used": False,
        }
        if any(warm.get(key) != value for key, value in required_warm.items()):
            raise RuntimeError("warm-start checkpoint is not a clean shared G1 PEFT package")
        if warm.get("capacity", {}).get("config") != capacity["config"]:
            raise RuntimeError("warm-start PEFT capacity differs from requested capacity")
        warm_provenance = warm.get("provenance", {})
        expected_warm_provenance = {
            "graph_sha256": sha256_file(args.graph),
            "hdf5_sha256": sha256_file(args.data),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
        }
        if any(
            warm_provenance.get(key) != value
            for key, value in expected_warm_provenance.items()
        ):
            raise RuntimeError("warm-start checkpoint provenance differs from current inputs")
        load_peft_state_dict(model, warm["peft_state"])
        warm_start_audit = {
            "checkpoint": str(args.initial_peft_checkpoint),
            "checkpoint_sha256": sha256_file(args.initial_peft_checkpoint),
            "source_best_epoch": int(warm.get("best_epoch", -1)),
            "source_formal": bool(warm.get("formal", False)),
            "source_chemical_supervision": False,
            "fallback_preserved_unless_validation_utility_strictly_improves": True,
        }
        initialization_kind = f"{initialization_kind}+clean_G1_PEFT_warm_start"
    initial_state = peft_state_dict(model)
    initial_state_sha256 = state_sha256(initial_state)
    prefix_cache_audit = {"enabled": False}
    if args.frozen_prefix_cache:
        raw_store = FrozenPrefixSpectrumStore(
            model, raw_store, device, args.prefix_cache_batch_size,
            last_blocks=args.last_blocks,
        )
        prefix_cache_audit = raw_store.audit
    teacher_store = None
    teacher_graph_embeddings = None
    teacher_graph_observable = None
    teacher_control_audit = None
    teacher_kind = None
    frozen_probe = None
    frozen_probe_fit_audit = None
    if args.molecule_teacher_dir is not None:
        teacher_store = MoleculeTeacherStore(
            args.molecule_teacher_dir, args.graph, graph, require_formal=formal
        )
        teacher_kind = str(
            teacher_store.report.get("teacher_kind", "molformer_connectivity")
        )
        (
            teacher_graph_embeddings,
            teacher_graph_observable,
            teacher_control_audit,
        ) = teacher_store.graph_embeddings(
            graph, allowed_molecule, str(args.teacher_control),
            args.seed + 1009 * args.outer_fold,
        )
        effective_queries = 0
        observable_negative_counts = []
        required_negatives = (
            1 if (
                args.chemical_objective in FROZEN_PROBE_OBJECTIVES
                or args.chemical_objective in {
                    "candidate_margin", "candidate_margin_targeted",
                }
            )
            else (2 if args.chemical_weighting == "relative_centered" else 1)
        )
        for query in train_query:
            left, right = map(int, graph.query_ptr[query:query + 2])
            selected = np.flatnonzero(allowed_molecule[left:right]) + left
            observed = teacher_graph_observable[selected]
            negatives = int(np.sum(observed[1:])) if bool(observed[0]) else 0
            observable_negative_counts.append(negatives)
            effective_queries += negatives >= required_negatives
        teacher_control_audit.update({
            "chemical_effect_queries": int(effective_queries),
            "chemical_effect_query_fraction": float(effective_queries / len(train_query)),
            "observable_negative_median": float(np.median(observable_negative_counts)),
            "teacher_observable_mask_sha256": hashlib.sha256(
                np.ascontiguousarray(teacher_graph_observable, dtype=np.bool_).tobytes()
            ).hexdigest(),
            "teacher_graph_embedding_sha256": hashlib.sha256(
                np.ascontiguousarray(teacher_graph_embeddings, dtype=np.float32).tobytes()
            ).hexdigest(),
        })
        if args.chemical_objective == "candidate_margin_targeted":
            teacher_control_audit.update(targeted_margin_query_audit(
                graph, train_query, allowed_molecule, teacher_graph_observable
            ))
        if args.chemical_objective in FROZEN_PROBE_OBJECTIVES:
            if teacher_kind != "morgan_binary_connectivity":
                raise RuntimeError("frozen-probe G2b currently requires the audited Morgan teacher")
            identity_targets, identity_observable, identity_audit = teacher_store.identity_targets(
                graph, allowed_molecule, str(args.teacher_control),
                args.seed + 1009 * args.outer_fold,
            )
            centroids = identity_reference_centroids(graph, official_store, teacher_store)
            fit_index = np.flatnonzero(identity_observable)
            fit = fit_frozen_ridge_probe(
                centroids[fit_index], identity_targets[fit_index], args.probe_alpha
            )
            frozen_probe = FrozenChemicalProbe(fit).to(device)
            if any(parameter.requires_grad for parameter in frozen_probe.parameters()):
                raise RuntimeError("frozen chemical probe unexpectedly has trainable parameters")
            selection = targeted_probe_query_audit(
                graph, train_query, allowed_molecule, teacher_graph_observable,
                args.probe_margin_threshold,
            )
            teacher_control_audit.update({
                key: value for key, value in selection.items()
                if key not in {"active_mask", "active_queries"}
            })
            effective_queries = int(selection["chemical_effect_queries"])
            frozen_probe_fit_audit = {
                "alpha": fit.alpha,
                "fit_identities": fit.examples,
                "input_dimension": int(fit.weight.shape[0]),
                "target_dimension": int(fit.weight.shape[1]),
                "teacher_identity_control": identity_audit,
                "input_mean_sha256": hashlib.sha256(
                    np.ascontiguousarray(fit.input_mean).tobytes()
                ).hexdigest(),
                "target_mean_sha256": hashlib.sha256(
                    np.ascontiguousarray(fit.target_mean).tobytes()
                ).hexdigest(),
                "weight_sha256": hashlib.sha256(
                    np.ascontiguousarray(fit.weight).tobytes()
                ).hexdigest(),
                "trainable_parameters": 0,
                "discarded_at_inference": True,
            }
        if effective_queries == 0:
            raise RuntimeError("molecule teacher changes no PEFT training candidate group")
    chemical_objective_name = (
        (
            f"training_only_peak_rule_attention_{args.peak_rule_bias_kind}_{args.peak_rule_objective}_{args.peak_rule_control}"
            if peak_rule_store is not None else None
        ) if teacher_store is None else (
            (
                "frozen_morgan_ridge_probe_targeted_multiview_listwise"
                if args.chemical_objective == "frozen_probe_targeted_multiview"
                else "frozen_morgan_ridge_probe_targeted_listwise"
            )
            if args.chemical_objective in FROZEN_PROBE_OBJECTIVES
            else (
                f"frozen_{teacher_kind}_{args.chemical_objective}"
                if args.chemical_objective in {"candidate_margin", "candidate_margin_targeted"}
                else f"frozen_{teacher_kind}_candidate_hardness_{args.chemical_weighting}"
            )
        )
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    official_encoded = np.asarray(official_store.official_embeddings, dtype=np.float32).copy()
    initial_encoded = encode_all(model, raw_store, device, args.eval_batch_size, args.amp)
    initial_cache_error = float(np.max(np.abs(initial_encoded - official_encoded)))
    initial_cosine = np.einsum("ij,ij->i", initial_encoded, official_encoded)
    initial_inner = paired_evaluation(
        initial_encoded, official_encoded, official_store, graph, inner_query
    )
    initial_peak_rule_view = None
    if peak_rule_store is not None:
        initial_peak_rule_encoded = encode_all(
            model, raw_store, device, args.eval_batch_size, args.amp,
            peak_rule_store=peak_rule_store,
        )
        initial_peak_rule_inner = paired_evaluation(
            initial_peak_rule_encoded, official_encoded, official_store, graph,
            inner_query,
        )
        initial_peak_rule_outer = paired_evaluation(
            initial_peak_rule_encoded, official_encoded, official_store, graph,
            outer_query,
        )
        initial_peak_rule_cosine = np.einsum(
            "ij,ij->i", initial_peak_rule_encoded, official_encoded
        )
        initial_peak_rule_view = {
            "inner": initial_peak_rule_inner["summary"],
            "outer": initial_peak_rule_outer["summary"],
            "official_embedding_cosine_mean": float(np.mean(initial_peak_rule_cosine)),
            "official_embedding_cosine_min": float(np.min(initial_peak_rule_cosine)),
            "direct_view_only": True,
            "used_for_model_selection": False,
            "discarded_at_inference": True,
        }
    if warm_start_audit is None:
        if (
            not np.array_equal(initial_inner["old_rank"], initial_inner["new_rank"])
            or initial_cache_error > 5e-4
            or float(np.min(initial_cosine)) < 0.9999
        ):
            raise RuntimeError(
                "zero-init PEFT does not reproduce official retrieval: "
                f"max_abs={initial_cache_error}, min_cosine={float(np.min(initial_cosine))}"
            )

    identities = graph.query_ik14[train_query].astype(str)
    weight_by_query, query_curriculum_audit = official_error_focus_weights(
        graph, train_query, allowed_molecule,
        args.official_error_focus_strength,
        args.official_error_focus_temperature,
    )
    initial_summary = initial_inner["summary"]
    initial_risk_net = (
        initial_summary["corrected"]
        - args.risk_penalty * initial_summary["introduced"]
    ) / initial_summary["n_queries"]
    initial_near_delta = initial_summary["delta_near_recall1"] or 0.0
    initial_utility = float(
        initial_risk_net
        + args.near_selection_weight * initial_near_delta
        + args.margin_selection_weight * initial_summary["delta_mean_margin"]
    )
    initial_eligible = model_selection_eligible(
        initial_summary,
        minimum_preservation=args.minimum_preservation,
        minimum_single_spectrum_preservation=args.minimum_single_spectrum_preservation,
        recall1_floor=args.recall1_floor,
        near_recall1_floor=args.near_recall1_floor,
    )
    if warm_start_audit is not None and not initial_eligible:
        raise RuntimeError("warm-start G1 checkpoint fails current model-selection safety floors")
    best_state = copy.deepcopy(initial_state)
    best_epoch = 0
    best_utility = initial_utility if initial_eligible else float("-inf")
    history = [{
        "epoch": 0, "inner": initial_inner["summary"],
        "risk_net_per_query": float(initial_risk_net),
        "selection_utility": initial_utility,
        "eligible": bool(initial_eligible),
    }]
    rng = np.random.default_rng(args.seed)
    first_step_audit = None
    positive_increment_gradient_audit = None
    print(
        f"[ChemAware-v3 {experiment} fold={args.outer_fold} seed={args.seed}] "
        f"epoch 0 inner={initial_inner['summary']}", flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.eval()
        order = rng.permutation(train_query)
        totals = {
            name: 0.0 for name in (
                "loss", "listwise", "molecule", "protect", "preserve",
                "positive_increment",
                "worst_preserve_hinge", "peak_rule_correct", "peak_rule_control",
                "peak_rule_active_fraction",
            )
        }
        molecule_total = pair_total = batches = 0
        chemical_gradient_audit = None
        chemical_gradient_attempts = 0
        started = time.time()
        for left in range(0, len(order), args.batch_queries):
            queries = order[left:left + args.batch_queries]
            batch = build_query_batch(graph, queries, allowed_molecule)
            (
                new_scores, old_scores, preserve, query_embeddings,
                positive_reference_embeddings, positive_reference_ptr,
                official_query_embeddings,
                official_positive_reference_embeddings,
                worst_preserve,
            ) = score_batch(
                model, raw_store, official_store, batch, device,
                args.forward_batch_size, args.amp,
            )
            query_ptr = torch.from_numpy(batch["query_ptr"]).to(device=device, dtype=torch.long)
            weights = torch.tensor(
                [weight_by_query[int(query)] for query in queries],
                device=device, dtype=torch.float32,
            )
            listwise = molecule_listwise_loss(
                new_scores, query_ptr, args.temperature, weights
            )
            protect = protected_margin_loss(
                new_scores, old_scores, query_ptr, args.protect_slack
            )
            worst_preserve_hinge = torch.relu(
                worst_preserve - args.worst_preserve_slack
            )
            positive_increment = positive_reference_increment_loss(
                query_embeddings,
                positive_reference_embeddings,
                positive_reference_ptr,
                official_query_embeddings,
                official_positive_reference_embeddings,
                increment=args.positive_increment,
                query_weights=weights,
                aggregation=args.positive_increment_aggregation,
            )
            if (
                args.lambda_positive_increment > 0
                and positive_increment_gradient_audit is None
            ):
                positive_increment_gradient_audit = gradient_geometry(
                    listwise, positive_increment, trainable
                )
                positive_increment_gradient_audit.update({
                    "objective": "same_identity_positive_reference_increment",
                    "aggregation": args.positive_increment_aggregation,
                    "increment": args.positive_increment,
                    "candidate_inputs_at_inference": False,
                })
                if (
                    positive_increment_gradient_audit[
                        "chemical_minus_clean_gradient_norm"
                    ] <= 1e-12
                ):
                    raise RuntimeError(
                        "positive-reference increment does not reach deployable parameters"
                    )
            peak_rule_correct_metric = new_scores.sum() * 0.0
            peak_rule_control_metric = new_scores.sum() * 0.0
            peak_rule_active_fraction = new_scores.sum() * 0.0
            if teacher_store is None:
                if peak_rule_store is None:
                    molecule = new_scores.sum() * 0.0
                    retrieval = listwise
                else:
                    peak_rule_scores = score_batch(
                        model, raw_store, official_store, batch, device,
                        args.forward_batch_size, args.amp,
                        peak_rule_store=peak_rule_store,
                    )[0]
                    peak_rule_correct_metric = molecule_listwise_loss(
                        peak_rule_scores, query_ptr, args.temperature, weights
                    )
                    if args.peak_rule_objective.startswith("matched_"):
                        if peak_rule_control_store is None:
                            raise RuntimeError("missing matched peak-rule control store")
                        peak_rule_control_scores = score_batch(
                            model, raw_store, official_store, batch, device,
                            args.forward_batch_size, args.amp,
                            peak_rule_store=peak_rule_control_store,
                        )[0]
                        peak_rule_control_metric = molecule_listwise_loss(
                            peak_rule_control_scores, query_ptr,
                            args.temperature, weights,
                        )
                        if args.peak_rule_objective == "matched_evidence_preference":
                            correct_values = molecule_listwise_loss_per_query(
                                peak_rule_scores, query_ptr, args.temperature
                            )
                            control_values = molecule_listwise_loss_per_query(
                                peak_rule_control_scores, query_ptr, args.temperature
                            )
                            active = (
                                (control_values - correct_values).detach()
                                > args.peak_rule_min_evidence
                            )
                            peak_rule_active_fraction = active.float().mean()
                            active_weights = weights * active.float()
                            if bool(torch.any(active)):
                                active_weights = active_weights / active_weights.sum().clamp_min(1e-12)
                                molecule = torch.sum(
                                    torch.relu(
                                        correct_values - control_values
                                        + args.peak_rule_preference_margin
                                    ) * active_weights
                                )
                            else:
                                molecule = (correct_values - control_values).sum() * 0.0
                        else:
                            molecule = torch.relu(
                                peak_rule_correct_metric
                                - peak_rule_control_metric
                                + args.peak_rule_preference_margin
                            )
                        retrieval = listwise
                        chemical_gradient_term = molecule
                    else:
                        molecule = peak_rule_correct_metric
                        retrieval = (
                            (1.0 - args.lambda_peak_rule) * listwise
                            + args.lambda_peak_rule * molecule
                        )
                        chemical_gradient_term = molecule - listwise
                    if chemical_gradient_audit is None and chemical_gradient_attempts < 10:
                        chemical_gradient_attempts += 1
                        candidate_audit = gradient_geometry(
                            listwise, chemical_gradient_term, trainable
                        )
                        candidate_audit.update({
                            "chemical_view": "peak_rule_attention",
                            "peak_rule_bias_kind": args.peak_rule_bias_kind,
                            "peak_rule_control": args.peak_rule_control,
                            "peak_rule_categories": list(args.peak_rule_categories),
                            "peak_rule_scale": args.peak_rule_view_scale,
                            "peak_rule_objective": args.peak_rule_objective,
                            "matched_control": (
                                "peak_permuted"
                                if peak_rule_control_store is not None else None
                            ),
                            "discarded_at_inference": True,
                        })
                        if candidate_audit["chemical_minus_clean_gradient_norm"] > 1e-12:
                            chemical_gradient_audit = candidate_audit
            else:
                teacher_values = torch.from_numpy(
                    teacher_graph_embeddings[batch["molecule_index"]]
                ).to(device=device, dtype=torch.float32)
                teacher_observable = torch.from_numpy(
                    teacher_graph_observable[batch["molecule_index"]]
                ).to(device=device, dtype=torch.bool)
                if args.chemical_objective in FROZEN_PROBE_OBJECTIVES:
                    if frozen_probe is None:
                        raise RuntimeError("missing fitted frozen chemical probe")
                    if args.chemical_objective == "frozen_probe_targeted_multiview":
                        molecule, active_probe_queries = (
                            targeted_probe_multiview_listwise_loss(
                                query_embeddings,
                                positive_reference_embeddings,
                                positive_reference_ptr,
                                old_scores,
                                teacher_values,
                                query_ptr,
                                frozen_probe,
                                args.probe_margin_threshold,
                                args.probe_temperature,
                                teacher_observable,
                                weights,
                            )
                        )
                    else:
                        molecule, active_probe_queries = targeted_probe_listwise_loss(
                            query_embeddings, old_scores, teacher_values, query_ptr,
                            frozen_probe, args.probe_margin_threshold,
                            args.probe_temperature, teacher_observable, weights,
                        )
                    retrieval = listwise
                elif args.chemical_objective == "candidate_margin_targeted":
                    molecule, active_probe_queries = targeted_chemical_margin_increment(
                        new_scores, old_scores, teacher_values, query_ptr,
                        args.molecule_temperature, args.chemical_margin_scale,
                        args.chemical_margin_similarity_threshold,
                        teacher_observable, weights,
                    )
                    retrieval = listwise + args.lambda_molecule * molecule
                elif args.chemical_objective == "candidate_margin":
                    molecule = chemical_margin_listwise_loss(
                        new_scores, teacher_values, query_ptr,
                        args.molecule_temperature, args.chemical_margin_scale,
                        teacher_observable, weights,
                    )
                    active_probe_queries = None
                    retrieval = (
                        (1.0 - args.lambda_molecule) * listwise
                        + args.lambda_molecule * molecule
                    )
                else:
                    molecule = chemical_weighted_listwise_loss(
                        new_scores, teacher_values, query_ptr,
                        args.molecule_temperature, args.chemical_hardness_beta,
                        teacher_observable, weights, args.chemical_weighting,
                    )
                    active_probe_queries = None
                    retrieval = (
                        (1.0 - args.lambda_molecule) * listwise
                        + args.lambda_molecule * molecule
                    )
                if chemical_gradient_audit is None and chemical_gradient_attempts < 10:
                    chemical_gradient_attempts += 1
                    candidate_audit = gradient_geometry(
                        listwise,
                        (
                            molecule
                            if (
                                args.chemical_objective in FROZEN_PROBE_OBJECTIVES
                                or args.chemical_objective == "candidate_margin_targeted"
                            )
                            else molecule - listwise
                        ),
                        trainable,
                    )
                    if active_probe_queries is not None:
                        candidate_audit["active_probe_queries_in_batch"] = int(
                            torch.sum(active_probe_queries).detach()
                        )
                        if args.chemical_objective == "candidate_margin_targeted":
                            candidate_audit["chemical_views"] = (
                                "official_wrong_candidate_scores_only"
                            )
                            candidate_audit["high_margin_correct_chemical_gradient"] = False
                        else:
                            candidate_audit["probe_views"] = (
                                "query_and_positive_references"
                                if args.chemical_objective
                                == "frozen_probe_targeted_multiview"
                                else "query_only"
                            )
                            candidate_audit["positive_reference_views_in_batch"] = int(
                                len(positive_reference_embeddings)
                            )
                    if candidate_audit["chemical_minus_clean_gradient_norm"] > 1e-12:
                        chemical_gradient_audit = candidate_audit
            loss = (
                args.lambda_listwise * retrieval
                + args.lambda_protect * protect
                + args.lambda_preserve * preserve
                + args.lambda_positive_increment * positive_increment
                + args.lambda_worst_preserve * worst_preserve_hinge
                + (
                    args.lambda_peak_rule * molecule
                    if peak_rule_store is not None
                    and args.peak_rule_objective.startswith("matched_")
                    else 0.0
                )
                + (
                    args.lambda_probe * molecule
                    if teacher_store is not None
                    and args.chemical_objective in FROZEN_PROBE_OBJECTIVES
                    else 0.0
                )
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite ChemAware v3 PEFT loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if first_step_audit is None:
                gradient_sq = sum(
                    float(torch.sum(parameter.grad.detach().double() ** 2))
                    for parameter in trainable if parameter.grad is not None
                )
                before_step = [parameter.detach().clone() for parameter in trainable]
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            if first_step_audit is None:
                update_sq = sum(
                    float(torch.sum((parameter.detach() - before).double() ** 2))
                    for parameter, before in zip(trainable, before_step)
                )
                first_step_audit = {
                    "gradient_l2": float(gradient_sq ** 0.5),
                    "parameter_update_l2": float(update_sq ** 0.5),
                    "changed_parameter_tensors": int(sum(
                        bool(torch.any(parameter.detach() != before))
                        for parameter, before in zip(trainable, before_step)
                    )),
                }
                if (
                    first_step_audit["gradient_l2"] <= 0
                    or first_step_audit["parameter_update_l2"] <= 0
                    or first_step_audit["changed_parameter_tensors"] <= 0
                ):
                    raise RuntimeError("PEFT first step did not reach or change deployable parameters")
            for name, value in (
                ("loss", loss), ("listwise", listwise), ("molecule", molecule),
                ("protect", protect), ("preserve", preserve),
                ("positive_increment", positive_increment),
                ("worst_preserve_hinge", worst_preserve_hinge),
                ("peak_rule_correct", peak_rule_correct_metric),
                ("peak_rule_control", peak_rule_control_metric),
                ("peak_rule_active_fraction", peak_rule_active_fraction),
            ):
                totals[name] += float(value.detach())
            molecule_total += len(batch["molecule_ptr"]) - 1
            pair_total += len(batch["candidate_rows"])
            batches += 1

        if (
            (teacher_store is not None or peak_rule_store is not None)
            and chemical_gradient_audit is None
        ):
            raise RuntimeError(
                "chemical view produced no incremental gradient on deployable PEFT parameters"
            )

        encoded = encode_all(model, raw_store, device, args.eval_batch_size, args.amp)
        inner = paired_evaluation(encoded, official_encoded, official_store, graph, inner_query)
        summary = inner["summary"]
        risk_net = (
            summary["corrected"] - args.risk_penalty * summary["introduced"]
        ) / summary["n_queries"]
        near_delta = summary["delta_near_recall1"] or 0.0
        utility = float(
            risk_net
            + args.near_selection_weight * near_delta
            + args.margin_selection_weight * summary["delta_mean_margin"]
        )
        eligible = model_selection_eligible(
            summary,
            minimum_preservation=args.minimum_preservation,
            minimum_single_spectrum_preservation=(
                args.minimum_single_spectrum_preservation
            ),
            recall1_floor=args.recall1_floor,
            near_recall1_floor=args.near_recall1_floor,
        )
        if eligible and utility > best_utility + 1e-12:
            best_state = copy.deepcopy(peft_state_dict(model))
            best_epoch = epoch
            best_utility = utility
        record = {
            "epoch": epoch,
            "train": {name: value / batches for name, value in totals.items()},
            "mean_candidate_molecules_per_batch": molecule_total / batches,
            "mean_candidate_spectrum_pairs_per_batch": pair_total / batches,
            "inner": summary,
            "risk_net_per_query": float(risk_net),
            "selection_utility": utility,
            "eligible": bool(eligible),
            "chemical_gradient_audit": chemical_gradient_audit,
            "seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"[ChemAware-v3 {experiment} fold={args.outer_fold} seed={args.seed}] {record}",
            flush=True,
        )

    load_peft_state_dict(model, best_state)
    encoded = encode_all(model, raw_store, device, args.eval_batch_size, args.amp)
    best_inner = paired_evaluation(
        encoded, official_encoded, official_store, graph, inner_query
    )
    outer = paired_evaluation(encoded, official_encoded, official_store, graph, outer_query)
    has_chemical_supervision = bool(
        teacher_store is not None or peak_rule_store is not None
    )
    output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "status": (
            "chemaware_shared_v3_peak_rule_peft"
            if peak_rule_store is not None else (
                "chemaware_shared_v3_clean_peft"
                if teacher_store is None else "chemaware_shared_v3_molecule_teacher_peft"
            )
        ),
        "format": "chemaware_shared_v3_peft_v1",
        "peft_state": best_state,
        "peft_config": capacity["config"],
        "capacity": capacity,
        "objective": (
            "complete_split_eligible_molecule_listwise_plus_positive_pair_increment"
            if args.lambda_positive_increment > 0
            else "complete_split_eligible_molecule_listwise"
        ),
        "chemical_supervision": has_chemical_supervision,
        "identity_positive_supervision": args.lambda_positive_increment > 0,
        "chemical_objective": chemical_objective_name,
        "training_only_projector_used": False,
        "training_only_frozen_probe_used": bool(frozen_probe is not None),
        "training_only_peak_rule_view_used": bool(peak_rule_store is not None),
        "chemical_gradient_absorber_trainable": False,
        "teacher_control": args.teacher_control,
        "teacher_kind": teacher_kind,
        "candidate_inputs_at_inference": False,
        "query_reference_encoder_shared": True,
        "P2b_used": False,
        "outer_fold": args.outer_fold,
        "inner_fold": inner_fold,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "formal": formal,
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "hdf5_sha256": sha256_file(args.data),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "raw_checkpoint_sha256": sha256_file(args.raw_checkpoint),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "molecule_teacher_report_sha256": (
                sha256_file(args.molecule_teacher_dir / "report.json")
                if args.molecule_teacher_dir is not None else None
            ),
            "script_sha256": sha256_file(Path(__file__)),
        },
    }, output / "peft.pt")
    np.savez_compressed(
        output / "outer_predictions.npz",
        query=outer["query"], old_rank=outer["old_rank"], new_rank=outer["new_rank"],
        old_margin=outer["old_margin"], new_margin=outer["new_margin"],
    )
    np.savez_compressed(
        output / "inner_predictions.npz",
        query=best_inner["query"], old_rank=best_inner["old_rank"],
        new_rank=best_inner["new_rank"], old_margin=best_inner["old_margin"],
        new_margin=best_inner["new_margin"],
    )
    decision = {
        "status": (
            "chemaware_shared_v3_g2_peft_fold_complete"
            if has_chemical_supervision else "chemaware_shared_v3_g1_peft_fold_complete"
        ),
        "formal": formal,
        "experiment": experiment,
        "chemical_supervision": has_chemical_supervision,
        "identity_positive_supervision": args.lambda_positive_increment > 0,
        "chemical_objective": chemical_objective_name,
        "training_only_projector_used": False,
        "training_only_frozen_probe_used": bool(frozen_probe is not None),
        "training_only_peak_rule_view_used": bool(peak_rule_store is not None),
        "chemical_gradient_absorber_trainable": False,
        "teacher_control": args.teacher_control,
        "teacher_kind": teacher_kind,
        "teacher_control_audit": teacher_control_audit,
        "peak_rule_attention_audit": peak_rule_audit,
        "frozen_probe_fit_audit": frozen_probe_fit_audit,
        "seed": args.seed,
        "outer_fold": args.outer_fold,
        "inner_fold": inner_fold,
        "train_queries": int(len(train_query)),
        "train_identities": int(len(np.unique(identities))),
        "query_curriculum_audit": query_curriculum_audit,
        "prefix_cache_audit": prefix_cache_audit,
        "warm_start_audit": warm_start_audit,
        "initial_peft_state_sha256": initial_state_sha256,
        "training_query_ledger_sha256": hashlib.sha256(train_query.tobytes()).hexdigest(),
        "allowed_molecule_ledger_sha256": hashlib.sha256(allowed_molecule.tobytes()).hexdigest(),
        "capacity": capacity,
        "training_contract": {
            "epochs": args.epochs,
            "batch_queries": args.batch_queries,
            "forward_batch_size": args.forward_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "lambda_listwise": args.lambda_listwise,
            "lambda_protect": args.lambda_protect,
            "lambda_preserve": args.lambda_preserve,
            "lambda_positive_increment": args.lambda_positive_increment,
            "positive_increment": args.positive_increment,
            "positive_increment_aggregation": args.positive_increment_aggregation,
            "lambda_worst_preserve": args.lambda_worst_preserve,
            "worst_preserve_slack": args.worst_preserve_slack,
            "protect_slack": args.protect_slack,
            "minimum_preservation": args.minimum_preservation,
            "minimum_single_spectrum_preservation": (
                args.minimum_single_spectrum_preservation
            ),
            "recall1_floor": args.recall1_floor,
            "near_recall1_floor": args.near_recall1_floor,
            "risk_penalty": args.risk_penalty,
            "near_selection_weight": args.near_selection_weight,
            "margin_selection_weight": args.margin_selection_weight,
            "official_error_focus_strength": args.official_error_focus_strength,
            "official_error_focus_temperature": (
                args.official_error_focus_temperature
            ),
            "frozen_prefix_cache": args.frozen_prefix_cache,
            "peak_rule_view_scale": args.peak_rule_view_scale,
            "peak_rule_bias_kind": args.peak_rule_bias_kind,
            "peak_rule_control": args.peak_rule_control,
            "peak_rule_categories": list(args.peak_rule_categories),
            "peak_rule_objective": args.peak_rule_objective,
            "peak_rule_preference_margin": args.peak_rule_preference_margin,
            "peak_rule_min_evidence": args.peak_rule_min_evidence,
            "lambda_peak_rule": args.lambda_peak_rule,
            "clean_g1_warm_start": args.initial_peft_checkpoint is not None,
            "folds": args.folds,
            "fold_seed": args.fold_seed,
            "dropout_during_peft": False,
        },
        "chemical_contract": (
            {
                "objective": chemical_objective_name,
                "lambda_peak_rule": args.lambda_peak_rule,
                "peak_rule_view_scale": args.peak_rule_view_scale,
                "peak_rule_bias_kind": args.peak_rule_bias_kind,
                "peak_rule_control": args.peak_rule_control,
                "categories": list(args.peak_rule_categories),
                "peak_rule_objective": args.peak_rule_objective,
                "preference_margin": args.peak_rule_preference_margin,
                "min_evidence": args.peak_rule_min_evidence,
                "combination": (
                    "clean_retrieval_plus_matched_correct_vs_peak_permuted_preference"
                    if args.peak_rule_objective.startswith("matched_")
                    else "convex_mix_of_clean_and_peak_rule_attention_retrieval"
                ),
                "spectrum_alignment_control": (
                    "paired_peak_permuted_in_same_batch"
                    if args.peak_rule_objective.startswith("matched_")
                    else "separate_control_run_required"
                ),
                "candidate_inputs_used": False,
                "training_only_view": True,
                "discarded_at_inference": True,
            }
            if peak_rule_store is not None else
            (
                {
                    "objective": args.chemical_objective,
                    "lambda_probe": args.lambda_probe,
                    "probe_alpha": args.probe_alpha,
                    "probe_temperature": args.probe_temperature,
                    "probe_margin_threshold": args.probe_margin_threshold,
                    "combination": "small_additive_loss_on_official_wrong_or_low_margin_queries",
                    "high_margin_official_correct_queries_receive_chemical_gradient": False,
                    "training_only_frozen_probe": True,
                    "frozen_probe_trainable_parameters": 0,
                    "training_only_projector_used": False,
                    "probe_views": (
                        "query_and_positive_references"
                        if args.chemical_objective == "frozen_probe_targeted_multiview"
                        else "query_only"
                    ),
                }
                if args.chemical_objective in FROZEN_PROBE_OBJECTIVES else
                {
                    "objective": args.chemical_objective,
                    "lambda_molecule": args.lambda_molecule,
                    "molecule_temperature": args.molecule_temperature,
                    "chemical_hardness_beta": args.chemical_hardness_beta,
                    "chemical_weighting": args.chemical_weighting,
                    "chemical_margin_scale": (
                        args.chemical_margin_scale
                        if args.chemical_objective in {
                            "candidate_margin", "candidate_margin_targeted",
                        } else None
                    ),
                    "chemical_margin_similarity_threshold": (
                        args.chemical_margin_similarity_threshold
                        if args.chemical_objective == "candidate_margin_targeted" else None
                    ),
                    "combination": (
                        "additive_margin_increment_on_frozen_official_errors"
                        if args.chemical_objective == "candidate_margin_targeted"
                        else "convex_mix_with_clean_listwise"
                    ),
                    "high_margin_official_correct_queries_receive_chemical_gradient": (
                        False
                        if args.chemical_objective == "candidate_margin_targeted"
                        else None
                    ),
                    "training_only_projector_used": False,
                }
            )
            if teacher_store is not None else None
        ),
        "initialization": {
            "kind": initialization_kind,
            "official_cache_max_abs_error": initial_cache_error,
            "official_cache_min_cosine": float(np.min(initial_cosine)),
            "initial_peak_rule_view": initial_peak_rule_view,
            "zero_init_exact_rank_reproduction": warm_start_audit is None,
            "warm_start_used": warm_start_audit is not None,
        },
        "first_step_audit": first_step_audit,
        "positive_increment_gradient_audit": positive_increment_gradient_audit,
        "best_epoch": best_epoch,
        "best_inner_utility": best_utility,
        "best_inner": best_inner["summary"],
        "outer": outer["summary"],
        "history": history,
        "preflight": {
            "exact_graph_coverage": True,
            "official_graph_max_abs_error": graph_score_error,
            "heldout_formula_molecules_excluded_from_training": True,
            "complete_split_eligible_candidate_groups": True,
        },
        "query_reference_encoder_shared": True,
        "candidate_inputs_at_inference": False,
        "P2b_used": False,
        "claim_limit": (
            (
                "G1 listwise plus same-identity positive-pair training control; one formula "
                "outer fold; no chemical-attribution claim and no sealed external claim"
                if args.lambda_positive_increment > 0
                else "clean PEFT capacity control only; one formula outer fold; no "
                "chemical-attribution claim and no sealed external claim"
            )
            if not has_chemical_supervision else
            "one G2 PEFT formula outer fold; chemical attribution requires paired multifold "
            "superiority over G1 and every pseudo-teacher; no sealed external claim"
        ),
    }
    json_dump(output / "decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
