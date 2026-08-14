"""CPU-budget sparse-autoencoder pilot for frozen DreaMS embeddings.

The current curated rule library is excluded from training and candidate selection.
This script tests whether an overcomplete Top-K sparse code yields reproducible
features that independently enrich exact spectral evidence and local structure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import combine_pvalues
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import discover_embedding_factors as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings", type=Path,
        default=ROOT / "data/validation/e0_baseline/e0_embeddings.npy",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/e0_baseline/e0_manifest.json",
    )
    parser.add_argument(
        "--hdf5", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/sparse_factor_discovery",
    )
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260811, 20260812, 20260813])
    parser.add_argument("--max-factors-to-annotate", type=int, default=128)
    parser.add_argument("--min-active", type=int, default=12)
    parser.add_argument("--max-active-fraction", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--component-whiten", action="store_true",
        help="Whiten PCA components separately (diagnostic only; off by default).",
    )
    return parser.parse_args()


class TiedTopKSAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, top_k: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.top_k = top_k
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        dense = F.relu(self.encoder(x))
        values, indices = torch.topk(dense, k=self.top_k, dim=1)
        sparse = torch.zeros_like(dense)
        sparse.scatter_(1, indices, values)
        return sparse

    def forward(self, x: torch.Tensor):
        code = self.encode(x)
        reconstruction = code @ self.encoder.weight + self.decoder_bias
        return reconstruction, code


def train_sae(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[TiedTopKSAE, np.ndarray, list[float], dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TiedTopKSAE(x_train.shape[1], args.hidden_dim, args.top_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_tensor = torch.from_numpy(x_train.astype(np.float32, copy=False))
    eval_tensor = torch.from_numpy(x_eval.astype(np.float32, copy=False))
    losses: list[float] = []

    model.train()
    for _ in range(args.epochs):
        order = rng.permutation(len(train_tensor))
        total = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            batch = train_tensor[ids]
            reconstruction, _ = model(batch)
            loss = F.mse_loss(reconstruction, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(ids)
            seen += len(ids)
        losses.append(total / seen)

    model.eval()
    with torch.no_grad():
        reconstruction, codes = model(eval_tensor)
        mse = float(F.mse_loss(reconstruction, eval_tensor))
        baseline = float(torch.mean(eval_tensor**2))
        codes_np = codes.numpy()
    metrics = {
        "eval_mse": mse,
        "zero_baseline_mse": baseline,
        "fraction_variance_unreconstructed": mse / baseline if baseline > 0 else float("nan"),
        "dead_feature_fraction": float(np.mean(np.sum(codes_np > 0, axis=0) == 0)),
        "median_activation_fraction": float(np.median(np.mean(codes_np > 0, axis=0))),
    }
    return model, codes_np, losses, metrics


def decoder_and_activation_stability(models, codes) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    reference = models[0].encoder.weight.detach().numpy()
    reference = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    decoder_similarities = []
    activation_correlations = []
    matches = []
    ref_codes = codes[0]
    for model, current_codes in zip(models[1:], codes[1:]):
        current = model.encoder.weight.detach().numpy()
        current = current / np.maximum(np.linalg.norm(current, axis=1, keepdims=True), 1e-12)
        similarity = reference @ current.T
        match = np.argmax(similarity, axis=1)
        matches.append(match)
        decoder_similarities.append(similarity[np.arange(len(reference)), match])
        corr = np.zeros(len(reference), dtype=float)
        for i, j in enumerate(match):
            a, b = ref_codes[:, i], current_codes[:, j]
            corr[i] = np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else 0.0
        activation_correlations.append(corr)
    return (
        np.mean(decoder_similarities, axis=0),
        np.mean(activation_correlations, axis=0),
        matches,
    )


def active_mask(scores: np.ndarray) -> np.ndarray:
    return scores > 0


def annotate_feature(
    scores: np.ndarray,
    morgan: np.ndarray,
    peaks: np.ndarray,
    losses: np.ndarray,
    molecules,
    generator,
) -> dict:
    mask = active_mask(scores)
    bit, bit_e, bit_a, bit_c, bit_pa, bit_pc = base.strongest_enrichment(morgan, mask, 4)
    peak, peak_e, peak_a, peak_c, _, _ = base.strongest_enrichment(peaks, mask, 4)
    loss, loss_e, loss_a, loss_c, _, _ = base.strongest_enrichment(losses, mask, 4)
    substructure = ""
    if bit >= 0:
        for idx in np.flatnonzero(mask):
            if molecules[idx] is not None and morgan[idx, bit]:
                substructure = base.environment_smiles(molecules[idx], generator, bit)
                if substructure:
                    break
    spectral_kind = "fragment_mz" if peak_e >= loss_e else "neutral_loss"
    return {
        "morgan_bit": bit,
        "morgan_log2_enrichment": bit_e,
        "morgan_support": bit_a,
        "morgan_background_support": bit_c,
        "morgan_top_prevalence": bit_pa,
        "morgan_background_prevalence": bit_pc,
        "substructure": substructure,
        "fragment_feature": peak,
        "fragment_log2_enrichment": peak_e,
        "fragment_support": peak_a,
        "fragment_background_support": peak_c,
        "loss_feature": loss,
        "loss_log2_enrichment": loss_e,
        "loss_support": loss_a,
        "loss_background_support": loss_c,
        "selected_spectral_kind": spectral_kind,
    }


def confirm_feature(
    scores: np.ndarray,
    discovery: dict,
    morgan: np.ndarray,
    peaks: np.ndarray,
    losses: np.ndarray,
) -> dict:
    mask = active_mask(scores)
    structure = base.selected_feature_enrichment(morgan, mask, discovery["morgan_bit"])
    structure_p = base.enrichment_p_value(morgan, mask, discovery["morgan_bit"])
    if discovery["selected_spectral_kind"] == "fragment_mz":
        matrix, feature = peaks, discovery["fragment_feature"]
    else:
        matrix, feature = losses, discovery["loss_feature"]
    spectral = base.selected_feature_enrichment(matrix, mask, feature)
    spectral_p = base.enrichment_p_value(matrix, mask, feature)
    combined = float(combine_pvalues([structure_p, spectral_p], method="fisher").pvalue)
    return {
        "active_count": int(mask.sum()),
        "structure": structure,
        "structure_p": structure_p,
        "spectral": spectral,
        "spectral_p": spectral_p,
        "combined_p": combined,
    }


def save_plot(catalog: pd.DataFrame, losses: list[list[float]], metrics: list[dict], path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, curve in enumerate(losses):
        axes[0].plot(np.arange(1, len(curve) + 1), curve, label=f"seed {i + 1}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training MSE")
    axes[0].set_title("Top-K SAE convergence")
    axes[0].legend(frameon=False)

    if len(catalog):
        axes[1].scatter(
            catalog["decoder_stability"], catalog["activation_stability"],
            c=catalog["confirmation_active_count"], cmap="viridis", s=28,
            edgecolor="black", linewidth=0.2,
        )
    else:
        axes[1].text(0.5, 0.5, "No eligible stable features", ha="center", va="center")
    axes[1].axvline(0.70, color="grey", ls="--", lw=1)
    axes[1].axhline(0.50, color="grey", ls="--", lw=1)
    axes[1].set_xlabel("Decoder direction stability")
    axes[1].set_ylabel("Activation stability")
    axes[1].set_title("Feature reproducibility across seeds")

    if len(catalog):
        axes[2].scatter(
            catalog["confirmation_structure_log2_enrichment"],
            catalog["confirmation_spectral_log2_enrichment"],
            c=catalog["combined_enrichment_q"], cmap="viridis_r", s=35,
            edgecolor="black", linewidth=0.2,
        )
    else:
        axes[2].text(0.5, 0.5, "No factors to annotate", ha="center", va="center")
    axes[2].axvline(1.0, color="grey", ls="--", lw=1)
    axes[2].axhline(1.0, color="grey", ls="--", lw=1)
    axes[2].set_xlabel("Independent structure enrichment (log2)")
    axes[2].set_ylabel("Independent exact-mass enrichment (log2)")
    axes[2].set_title("Chemical evidence on confirmation molecules")

    fig.suptitle("Sparse latent factors from frozen raw-SSL DreaMS embeddings", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = base.choose_one_spectrum_per_molecule(manifest)
    embeddings = np.load(args.embeddings, mmap_mode="r")
    x = np.asarray(embeddings[selected["embedding_idx"].to_numpy(int)], dtype=np.float32)

    train_idx, heldout_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seeds[0]).split(
            x, groups=selected["inchikey_14"]
        )
    )
    heldout = selected.iloc[heldout_idx].reset_index(drop=True)
    annotation_idx, confirmation_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=args.seeds[0] + 17).split(
            heldout, groups=heldout["inchikey_14"]
        )
    )

    reducer = PCA(n_components=args.pca_dim, svd_solver="randomized", random_state=args.seeds[0])
    train_reduced = reducer.fit_transform(x[train_idx])
    heldout_reduced = reducer.transform(x[heldout_idx])
    if args.component_whiten:
        center = train_reduced.mean(axis=0, keepdims=True)
        scale = train_reduced.std(axis=0, keepdims=True)
        scale = np.maximum(scale, 1e-8)
        train_sae_input = ((train_reduced - center) / scale).astype(np.float32)
        heldout_sae_input = ((heldout_reduced - center) / scale).astype(np.float32)
        normalization = "per-component whitening"
    else:
        # Preserve the variance geometry of the frozen embedding.  Whitening every
        # PCA axis makes the data nearly rotationally symmetric and causes equally
        # valid SAEs to choose incompatible bases across random seeds.
        center = train_reduced.mean(axis=0, keepdims=True)
        scale = float(np.sqrt(np.mean((train_reduced - center) ** 2)))
        train_sae_input = ((train_reduced - center) / scale).astype(np.float32)
        heldout_sae_input = ((heldout_reduced - center) / scale).astype(np.float32)
        normalization = "global RMS scaling; PCA variance geometry preserved"

    models, heldout_codes, loss_curves, sae_metrics = [], [], [], []
    for seed in args.seeds:
        model, codes, curve, metrics = train_sae(train_sae_input, heldout_sae_input, args, seed)
        models.append(model)
        heldout_codes.append(codes)
        loss_curves.append(curve)
        sae_metrics.append(metrics)
        torch.save(model.state_dict(), args.output_dir / f"topk_sae_seed_{seed}.pt")

    decoder_stability, activation_stability, _ = decoder_and_activation_stability(models, heldout_codes)
    reference_codes = heldout_codes[0]
    annotation_counts = np.sum(reference_codes[annotation_idx] > 0, axis=0)
    confirmation_counts = np.sum(reference_codes[confirmation_idx] > 0, axis=0)
    stability_frame = pd.DataFrame({
        "factor": np.arange(args.hidden_dim, dtype=int),
        "decoder_stability": decoder_stability,
        "activation_stability": activation_stability,
        "annotation_active_count": annotation_counts,
        "confirmation_active_count": confirmation_counts,
        "heldout_activation_fraction": np.mean(reference_codes > 0, axis=0),
    })
    stability_frame.to_csv(args.output_dir / "all_feature_stability.csv", index=False)
    max_active = int(math.floor(len(annotation_idx) * args.max_active_fraction))
    eligible = (
        (decoder_stability >= 0.70)
        & (activation_stability >= 0.50)
        & (annotation_counts >= args.min_active)
        & (confirmation_counts >= args.min_active)
        & (annotation_counts <= max_active)
        & (confirmation_counts <= max_active)
    )
    priority = np.lexsort((-activation_stability, -decoder_stability))
    factor_ids = [int(i) for i in priority if eligible[i]][:args.max_factors_to_annotate]

    peaks, losses = base.make_presence_matrices(args.hdf5, heldout)
    morgan, molecules, generator = base.make_morgan_matrix(heldout["smiles"].tolist())
    annotation_molecules = [molecules[i] for i in annotation_idx]

    precursor = heldout["precursor_mz"].to_numpy(float)
    peak_count = heldout["n_peaks"].to_numpy(float)
    collision_energy = heldout["ce"].to_numpy(float)
    instrument = heldout["instrument"].fillna("unknown").to_numpy(str)

    rows = []
    for factor in factor_ids:
        discovery = annotate_feature(
            reference_codes[annotation_idx, factor],
            morgan[annotation_idx], peaks[annotation_idx], losses[annotation_idx],
            annotation_molecules, generator,
        )
        confirmation = confirm_feature(
            reference_codes[confirmation_idx, factor], discovery,
            morgan[confirmation_idx], peaks[confirmation_idx], losses[confirmation_idx],
        )
        scores = reference_codes[:, factor]
        confounds = [
            base.safe_abs_corr(scores, precursor),
            base.safe_abs_corr(scores, peak_count),
            base.safe_abs_corr(scores, collision_energy),
            base.eta_squared(scores, instrument),
        ]
        max_confound = float(np.nanmax(confounds))
        spectral_feature = (
            discovery["fragment_feature"]
            if discovery["selected_spectral_kind"] == "fragment_mz"
            else discovery["loss_feature"]
        )
        if spectral_feature < 0:
            spectral_da = float("nan")
        elif discovery["selected_spectral_kind"] == "fragment_mz":
            spectral_da = (spectral_feature + 0.5) * base.MZ_BIN_WIDTH
        else:
            spectral_da = (spectral_feature + 0.5) * base.LOSS_BIN_WIDTH
        rows.append({
            "factor": factor,
            "decoder_stability": decoder_stability[factor],
            "activation_stability": activation_stability[factor],
            "annotation_active_count": int(annotation_counts[factor]),
            "confirmation_active_count": int(confirmation_counts[factor]),
            "activation_fraction_all_heldout": float(np.mean(scores > 0)),
            "max_confound_association": max_confound,
            "morgan_bit": discovery["morgan_bit"],
            "representative_substructure": discovery["substructure"],
            "selected_spectral_kind": discovery["selected_spectral_kind"],
            "selected_spectral_mass_Da": spectral_da,
            "confirmation_structure_log2_enrichment": confirmation["structure"]["log2_enrichment"],
            "confirmation_structure_support": confirmation["structure"]["top_support"],
            "confirmation_structure_p": confirmation["structure_p"],
            "confirmation_spectral_log2_enrichment": confirmation["spectral"]["log2_enrichment"],
            "confirmation_spectral_support": confirmation["spectral"]["top_support"],
            "confirmation_spectral_p": confirmation["spectral_p"],
            "combined_enrichment_p": confirmation["combined_p"],
        })

    catalog = pd.DataFrame(rows)
    if len(catalog):
        catalog["combined_enrichment_q"] = base.benjamini_hochberg(
            catalog["combined_enrichment_p"].to_numpy(float)
        )
        catalog["candidate_for_occlusion"] = (
            (catalog["confirmation_structure_log2_enrichment"] >= 1.0)
            & (catalog["confirmation_spectral_log2_enrichment"] >= 1.0)
            & (catalog["confirmation_structure_support"] >= 4)
            & (catalog["confirmation_spectral_support"] >= 4)
            & (catalog["confirmation_structure_p"] <= 0.05)
            & (catalog["confirmation_spectral_p"] <= 0.05)
            & (catalog["combined_enrichment_q"] <= 0.10)
            & (catalog["max_confound_association"] < 0.50)
        )
    else:
        catalog = pd.DataFrame(columns=["factor", "combined_enrichment_q", "candidate_for_occlusion"])
    catalog.to_csv(args.output_dir / "sparse_factor_catalog.csv", index=False)

    save_plot(catalog, loss_curves, sae_metrics, args.output_dir / "sparse_factor_overview.png")
    summary = {
        "checkpoint": "ssl_model_server.pt (raw self-supervised DreaMS; cached embeddings)",
        "unique_molecules": len(selected),
        "train_molecules": len(train_idx),
        "annotation_molecules": len(annotation_idx),
        "confirmation_molecules": len(confirmation_idx),
        "pca_dim": args.pca_dim,
        "pca_variance_retained": float(np.sum(reducer.explained_variance_ratio_)),
        "normalization": normalization,
        "hidden_dim": args.hidden_dim,
        "top_k": args.top_k,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "sae_metrics": sae_metrics,
        "decoder_stability_quantiles": {
            str(q): float(np.quantile(decoder_stability, q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "activation_stability_quantiles": {
            str(q): float(np.quantile(activation_stability, q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "eligible_stable_features": int(np.sum(eligible)),
        "annotated_features": len(factor_ids),
        "candidate_features_for_occlusion": (
            int(catalog["candidate_for_occlusion"].sum()) if len(catalog) else 0
        ),
        "candidate_factor_ids": (
            catalog.loc[catalog["candidate_for_occlusion"], "factor"].astype(int).tolist()
            if len(catalog) else []
        ),
    }
    (args.output_dir / "sae_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# DreaMS 稀疏潜在因子 CPU 预实验",
        "",
        f"- 权重来源：原始自监督 `ssl_model_server.pt` 的缓存 precursor embedding。",
        f"- 分子：训练 {len(train_idx):,}；候选注释 {len(annotation_idx):,}；独立确认 {len(confirmation_idx):,}。",
        f"- 输入压缩：1024→{args.pca_dim}，保留方差 {np.sum(reducer.explained_variance_ratio_):.3f}。",
        f"- 归一化：{normalization}。",
        f"- SAE：{args.pca_dim}→{args.hidden_dim}，Top-K={args.top_k}，{len(args.seeds)} 个随机种子，各 {args.epochs} epochs。",
        f"- 稳定且激活支持充足的特征：{int(np.sum(eligible))}；进入独立注释：{len(factor_ids)}。",
        f"- 进入峰遮蔽验证的候选：{summary['candidate_features_for_occlusion']}。",
        "",
        "候选因子必须在独立分子中同时复现局部结构和 0.02 Da 谱图证据，并通过混杂检查和多重检验校正。",
    ]
    if len(catalog):
        top = catalog.sort_values(
            ["candidate_for_occlusion", "combined_enrichment_q", "decoder_stability"],
            ascending=[False, True, False],
        ).head(12)
        report += [
            "",
            "| 因子 | 解码器稳定性 | 激活稳定性 | 结构富集 | 谱图富集 | BH q | 结构片段 | 精确质量 | 候选 |",
            "|---:|---:|---:|---:|---:|---:|---|---:|---|",
        ]
        for row in top.itertuples():
            report.append(
                f"| S{int(row.factor):04d} | {row.decoder_stability:.3f} | {row.activation_stability:.3f} | "
                f"{row.confirmation_structure_log2_enrichment:.2f} | {row.confirmation_spectral_log2_enrichment:.2f} | "
                f"{row.combined_enrichment_q:.3g} | `{row.representative_substructure}` | "
                f"{row.selected_spectral_mass_Da:.4f} | {bool(row.candidate_for_occlusion)} |"
            )
    (args.output_dir / "SAE_FACTOR_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(catalog):
        print(catalog.sort_values("combined_enrichment_q").head(15).to_string(index=False))


if __name__ == "__main__":
    main()
