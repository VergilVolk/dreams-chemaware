"""Discover latent chemical factors from frozen DreaMS embeddings.

This pilot deliberately excludes the project's curated chemical-rule vectors from
factor fitting and selection.  Factors are learned from precursor embeddings only,
then annotated on held-out molecules with data-derived spectral bins, neutral-loss
bins and Morgan atom-environment bits.  The annotations are descriptive and do not
turn a latent factor into a proven fragmentation mechanism.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator
from scipy.optimize import linear_sum_assignment
from scipy.stats import combine_pvalues, fisher_exact
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
MZ_BIN_WIDTH = 0.02
LOSS_BIN_WIDTH = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "data/validation/e0_baseline/e0_embeddings.npy",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/validation/e0_baseline/e0_manifest.json",
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/validation/embedding_factor_discovery",
    )
    parser.add_argument("--n-factors", type=int, default=32)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--stability-runs", type=int, default=5)
    parser.add_argument("--stability-subsample", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def varimax(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 200, tol: float = 1e-6):
    """Orthogonal varimax rotation, returning rotated loadings and rotation."""
    n_features, n_factors = loadings.shape
    rotation = np.eye(n_factors)
    previous = 0.0
    for _ in range(max_iter):
        rotated = loadings @ rotation
        u, singular, vt = np.linalg.svd(
            loadings.T
            @ (rotated**3 - (gamma / n_features) * rotated @ np.diag(np.sum(rotated**2, axis=0))),
            full_matrices=False,
        )
        rotation = u @ vt
        objective = float(np.sum(singular))
        if previous and objective / previous < 1.0 + tol:
            break
        previous = objective
    return loadings @ rotation, rotation


def fit_factor_model(x: np.ndarray, n_factors: int, seed: int):
    pca = PCA(n_components=n_factors, svd_solver="randomized", random_state=seed)
    pca.fit(x)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)[None, :]
    rotated_loadings, rotation = varimax(loadings)
    return pca, rotated_loadings, rotation


def factor_scores(pca: PCA, rotation: np.ndarray, x: np.ndarray) -> np.ndarray:
    return pca.transform(x) @ rotation


def choose_one_spectrum_per_molecule(manifest: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(manifest)
    frame = frame.loc[frame["inchikey_14"].notna() & frame["smiles"].notna()].copy()
    frame["n_peaks"] = pd.to_numeric(frame["n_peaks"], errors="coerce").fillna(-1)
    frame["ce"] = pd.to_numeric(frame["ce"], errors="coerce")
    frame["precursor_mz"] = pd.to_numeric(frame["precursor_mz"], errors="coerce")
    # A deterministic, information-rich representative avoids replicate-rich molecules
    # dominating factor discovery while retaining enough peaks for later annotation.
    frame = frame.sort_values(
        ["inchikey_14", "n_peaks", "spectrum_id"], ascending=[True, False, True]
    )
    return frame.drop_duplicates("inchikey_14", keep="first").reset_index(drop=True)


def loading_stability(
    x_train: np.ndarray,
    reference_loadings: np.ndarray,
    n_factors: int,
    runs: int,
    subsample_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ref = reference_loadings / np.maximum(np.linalg.norm(reference_loadings, axis=0), 1e-12)
    matched = []
    n_sub = max(n_factors + 5, int(len(x_train) * subsample_fraction))
    for run in range(runs):
        ids = rng.choice(len(x_train), size=n_sub, replace=False)
        _, loadings, _ = fit_factor_model(x_train[ids], n_factors, seed + run + 1)
        cur = loadings / np.maximum(np.linalg.norm(loadings, axis=0), 1e-12)
        similarity = np.abs(ref.T @ cur)
        rows, cols = linear_sum_assignment(-similarity)
        values = np.zeros(n_factors, dtype=float)
        values[rows] = similarity[rows, cols]
        matched.append(values)
    values = np.asarray(matched)
    return values.mean(axis=0), values.min(axis=0)


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    mask = np.isfinite(values) & pd.notna(groups)
    values = values[mask]
    groups = groups[mask]
    if len(values) < 3 or np.var(values) <= 0:
        return float("nan")
    mean = float(np.mean(values))
    between = 0.0
    for group in np.unique(groups):
        part = values[groups == group]
        between += len(part) * float((np.mean(part) - mean) ** 2)
    total = float(np.sum((values - mean) ** 2))
    return between / total if total > 0 else float("nan")


def safe_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) <= 0 or np.std(b[mask]) <= 0:
        return float("nan")
    return abs(float(np.corrcoef(a[mask], b[mask])[0, 1]))


def make_presence_matrices(
    hdf5_path: Path,
    selected: pd.DataFrame,
    mz_width: float = MZ_BIN_WIDTH,
    loss_width: float = LOSS_BIN_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as h5:
        identifiers = [decode(v) for v in h5["IDENTIFIER"][:]]
        index = {identifier: i for i, identifier in enumerate(identifiers)}
        rows = np.asarray([index[sid] for sid in selected["spectrum_id"]], dtype=int)
        # h5py fancy indexing requires monotonically increasing indices. Read in
        # file order, then restore the selected dataframe order.
        order = np.argsort(rows)
        sorted_spectra = h5["spectrum"][rows[order]]
        spectra = np.empty_like(sorted_spectra)
        spectra[order] = sorted_spectra

    peak_bins = np.zeros((len(selected), int(math.ceil(1000 / mz_width))), dtype=bool)
    loss_bins = np.zeros((len(selected), int(math.ceil(500 / loss_width))), dtype=bool)
    precursor = selected["precursor_mz"].to_numpy(float)
    for i, spectrum in enumerate(spectra):
        mz = np.asarray(spectrum[0], dtype=float)
        intensity = np.asarray(spectrum[1], dtype=float)
        # Exclude the precursor/isotope vicinity from fragment-factor annotation;
        # otherwise an apparent "fragment" factor can simply encode precursor mass.
        valid = (
            (mz > 0)
            & (intensity > 0)
            & np.isfinite(mz)
            & (mz < precursor[i] - 1.5)
        )
        mz = mz[valid]
        bins = np.floor(mz / mz_width).astype(int)
        bins = bins[(bins >= 0) & (bins < peak_bins.shape[1])]
        peak_bins[i, np.unique(bins)] = True

        losses = precursor[i] - mz
        lbins = np.floor(losses / loss_width).astype(int)
        lbins = lbins[(lbins >= 0) & (lbins < loss_bins.shape[1])]
        loss_bins[i, np.unique(lbins)] = True
    return peak_bins, loss_bins


def make_morgan_matrix(smiles: list[str], n_bits: int = 2048):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    molecules: list[Chem.Mol | None] = []
    matrix = np.zeros((len(smiles), n_bits), dtype=bool)
    for i, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(text)
        molecules.append(mol)
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        matrix[i, list(fp.GetOnBits())] = True
    return matrix, molecules, generator


def strongest_enrichment(presence: np.ndarray, selected_mask: np.ndarray, min_top_support: int = 4):
    top = presence[selected_mask]
    background = presence[~selected_mask]
    a = top.sum(axis=0).astype(float)
    c = background.sum(axis=0).astype(float)
    top_prev = (a + 0.5) / (len(top) + 1.0)
    bg_prev = (c + 0.5) / (len(background) + 1.0)
    log2_enrichment = np.log2(top_prev / bg_prev)
    valid = a >= min_top_support
    if not np.any(valid):
        return -1, float("nan"), 0, 0, float("nan"), float("nan")
    score = np.where(valid, log2_enrichment, -np.inf)
    idx = int(np.argmax(score))
    return idx, float(score[idx]), int(a[idx]), int(c[idx]), float(top_prev[idx]), float(bg_prev[idx])


def pole_mask(score: np.ndarray, pole: str, top_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    n_top = max(20, int(math.ceil(len(score) * top_fraction)))
    order = np.argsort(score)
    top_ids = order[-n_top:] if pole == "positive" else order[:n_top]
    mask = np.zeros(len(score), dtype=bool)
    mask[top_ids] = True
    return mask, top_ids


def selected_feature_enrichment(presence: np.ndarray, selected_mask: np.ndarray, feature: int) -> dict:
    """Evaluate one preselected feature without searching over alternatives."""
    if feature < 0:
        return {
            "log2_enrichment": float("nan"), "top_support": 0,
            "background_support": 0, "top_prevalence": float("nan"),
            "background_prevalence": float("nan"),
        }
    top = presence[selected_mask, feature]
    background = presence[~selected_mask, feature]
    a = int(top.sum())
    c = int(background.sum())
    top_prev = a / len(top) if len(top) else float("nan")
    bg_prev = c / len(background) if len(background) else float("nan")
    if a == 0:
        log2_enrichment = 0.0 if c == 0 else float("-inf")
    elif c == 0:
        # A finite lower-bound estimate is useful when the selected feature occurs
        # in the active group but is absent from a much larger background group.
        log2_enrichment = float(np.log2(top_prev / (0.5 / (len(background) + 1.0))))
    else:
        log2_enrichment = float(np.log2(top_prev / bg_prev))
    return {
        "log2_enrichment": log2_enrichment,
        "top_support": a,
        "background_support": c,
        "top_prevalence": float(top_prev),
        "background_prevalence": float(bg_prev),
    }


def enrichment_p_value(presence: np.ndarray, selected_mask: np.ndarray, feature: int) -> float:
    if feature < 0:
        return float("nan")
    top = presence[selected_mask, feature]
    background = presence[~selected_mask, feature]
    table = [
        [int(top.sum()), int((~top).sum())],
        [int(background.sum()), int((~background).sum())],
    ]
    return float(fisher_exact(table, alternative="greater").pvalue)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if not len(valid):
        return q_values
    order = valid[np.argsort(p_values[valid])]
    ranked = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values[order] = np.clip(ranked, 0.0, 1.0)
    return q_values


def environment_smiles(mol: Chem.Mol, generator, bit: int) -> str:
    output = rdFingerprintGenerator.AdditionalOutput()
    output.AllocateBitInfoMap()
    generator.GetFingerprint(mol, additionalOutput=output)
    info = output.GetBitInfoMap().get(int(bit), ())
    if not info:
        return ""
    atom_idx, radius = info[0]
    if radius == 0:
        return f"[{mol.GetAtomWithIdx(atom_idx).GetSymbol()}]"
    bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, int(radius), int(atom_idx))
    atoms = {int(atom_idx)}
    for bond_idx in bonds:
        bond = mol.GetBondWithIdx(int(bond_idx))
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    fragment = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), bondsToUse=list(bonds), canonical=True)
    return fragment


def annotate_pole(
    score: np.ndarray,
    pole: str,
    top_fraction: float,
    morgan: np.ndarray,
    peak_presence: np.ndarray,
    loss_presence: np.ndarray,
    molecules: list[Chem.Mol | None],
    generator,
) -> dict:
    mask, top_ids = pole_mask(score, pole, top_fraction)
    n_top = len(top_ids)

    bit, bit_enrich, bit_top, bit_bg, bit_top_prev, bit_bg_prev = strongest_enrichment(morgan, mask)
    peak, peak_enrich, peak_top, peak_bg, _, _ = strongest_enrichment(peak_presence, mask)
    loss, loss_enrich, loss_top, loss_bg, _, _ = strongest_enrichment(loss_presence, mask)

    substructure = ""
    if bit >= 0:
        for idx in top_ids:
            if molecules[idx] is not None and morgan[idx, bit]:
                substructure = environment_smiles(molecules[idx], generator, bit)
                if substructure:
                    break

    return {
        "pole": pole,
        "n_top": n_top,
        "top_ids": top_ids,
        "morgan_bit": bit,
        "morgan_log2_enrichment": bit_enrich,
        "morgan_top_support": bit_top,
        "morgan_background_support": bit_bg,
        "morgan_top_prevalence": bit_top_prev,
        "morgan_background_prevalence": bit_bg_prev,
        "representative_substructure": substructure,
        "fragment_mz_bin": peak,
        "fragment_log2_enrichment": peak_enrich,
        "fragment_top_support": peak_top,
        "fragment_background_support": peak_bg,
        "neutral_loss_bin": loss,
        "neutral_loss_log2_enrichment": loss_enrich,
        "neutral_loss_top_support": loss_top,
        "neutral_loss_background_support": loss_bg,
    }


def save_overview(catalog: pd.DataFrame, explained: np.ndarray, path: Path):
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(np.arange(1, len(explained) + 1), np.cumsum(explained), marker="o", ms=3)
    axes[0].set_xlabel("Number of PCA components")
    axes[0].set_ylabel("Cumulative explained variance")
    axes[0].set_title("Embedding variance retained")

    scatter = axes[1].scatter(
        catalog["stability_mean"],
        catalog["best_morgan_log2_enrichment"],
        c=catalog["max_confound_association"],
        s=35 + 15 * np.clip(catalog["best_spectral_log2_enrichment"], 0, 5),
        cmap="viridis_r",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[1].axvline(0.70, color="grey", ls="--", lw=1)
    axes[1].set_xlabel("Resampling stability (loading cosine)")
    axes[1].set_ylabel("Held-out structure enrichment (log2)")
    axes[1].set_title("Stable and chemically enriched factors")
    fig.colorbar(scatter, ax=axes[1], label="Largest confound association")

    ranked = catalog.sort_values(
        ["candidate_for_review", "stability_mean", "best_morgan_log2_enrichment"],
        ascending=False,
    ).head(10)
    y = np.arange(len(ranked))
    axes[2].barh(y - 0.18, ranked["best_morgan_log2_enrichment"], height=0.35, label="Structure")
    axes[2].barh(y + 0.18, ranked["best_spectral_log2_enrichment"], height=0.35, label="Peak/loss")
    axes[2].set_yticks(y, [f"F{int(v):02d}" for v in ranked["factor"]])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Best held-out enrichment (log2)")
    axes[2].set_title("Factors prioritized for review")
    axes[2].legend(frameon=False)

    fig.suptitle("Rule-free discovery of latent factors in frozen DreaMS embeddings", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_molecule_grid(catalog: pd.DataFrame, examples: pd.DataFrame, path: Path):
    chosen = catalog.sort_values(
        ["candidate_for_review", "stability_mean", "best_morgan_log2_enrichment"],
        ascending=False,
    ).head(6)
    mols, legends = [], []
    for factor in chosen["factor"]:
        rows = examples.loc[examples["factor"] == factor].sort_values("rank_within_pole").head(3)
        for row in rows.itertuples():
            mol = Chem.MolFromSmiles(row.smiles)
            if mol is not None:
                mols.append(mol)
                legends.append(f"F{int(factor):02d} {row.pole}\nscore={row.factor_score:.2f}")
    if mols:
        image = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=3, subImgSize=(320, 240))
        image.save(str(path))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_embeddings = np.load(args.embeddings, mmap_mode="r")
    selected = choose_one_spectrum_per_molecule(manifest)
    x = np.asarray(all_embeddings[selected["embedding_idx"].to_numpy(int)], dtype=np.float32)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.holdout_fraction, random_state=args.seed)
    train_idx, holdout_idx = next(splitter.split(x, groups=selected["inchikey_14"]))
    x_train = x[train_idx]
    x_holdout = x[holdout_idx]
    holdout = selected.iloc[holdout_idx].reset_index(drop=True)

    pca, loadings, rotation = fit_factor_model(x_train, args.n_factors, args.seed)
    holdout_scores = factor_scores(pca, rotation, x_holdout)
    stability_mean, stability_min = loading_stability(
        x_train,
        loadings,
        args.n_factors,
        args.stability_runs,
        args.stability_subsample,
        args.seed,
    )

    peak_presence, loss_presence = make_presence_matrices(args.hdf5, holdout)
    morgan, molecules, generator = make_morgan_matrix(holdout["smiles"].tolist())

    # The first half of the held-out molecules is allowed to suggest a factor's
    # meaning.  The second half only evaluates those already-selected features.
    # This prevents a maximum over thousands of bins/bits being reported as if it
    # were an independent validation result.
    annotation_idx, confirmation_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=args.seed + 17).split(
            holdout, groups=holdout["inchikey_14"]
        )
    )
    annotation_molecules = [molecules[i] for i in annotation_idx]

    precursor = holdout["precursor_mz"].to_numpy(float)
    peak_count = holdout["n_peaks"].to_numpy(float)
    collision_energy = holdout["ce"].to_numpy(float)
    instrument = holdout["instrument"].fillna("unknown").to_numpy(str)

    catalog_rows = []
    example_rows = []
    for factor in range(args.n_factors):
        score = holdout_scores[:, factor]
        annotation_score = score[annotation_idx]
        positive = annotate_pole(
            annotation_score,
            "positive",
            args.top_fraction,
            morgan[annotation_idx],
            peak_presence[annotation_idx],
            loss_presence[annotation_idx],
            annotation_molecules,
            generator,
        )
        negative = annotate_pole(
            annotation_score,
            "negative",
            args.top_fraction,
            morgan[annotation_idx],
            peak_presence[annotation_idx],
            loss_presence[annotation_idx],
            annotation_molecules,
            generator,
        )
        poles = [positive, negative]
        discovery = max(
            poles,
            key=lambda row: (
                np.nan_to_num(row["morgan_log2_enrichment"], nan=-np.inf)
                + np.nan_to_num(
                    max(row["fragment_log2_enrichment"], row["neutral_loss_log2_enrichment"]),
                    nan=-np.inf,
                )
            ),
        )

        confirmation_score = score[confirmation_idx]
        confirmation_mask, confirmation_top_ids = pole_mask(
            confirmation_score, discovery["pole"], args.top_fraction
        )
        confirmed_structure = selected_feature_enrichment(
            morgan[confirmation_idx], confirmation_mask, discovery["morgan_bit"]
        )
        confirmed_fragment = selected_feature_enrichment(
            peak_presence[confirmation_idx], confirmation_mask, discovery["fragment_mz_bin"]
        )
        confirmed_loss = selected_feature_enrichment(
            loss_presence[confirmation_idx], confirmation_mask, discovery["neutral_loss_bin"]
        )
        selected_spectral_kind = (
            "fragment_mz"
            if discovery["fragment_log2_enrichment"] >= discovery["neutral_loss_log2_enrichment"]
            else "neutral_loss"
        )
        confirmed_spectral = (
            confirmed_fragment if selected_spectral_kind == "fragment_mz" else confirmed_loss
        )
        structure_p = enrichment_p_value(
            morgan[confirmation_idx], confirmation_mask, discovery["morgan_bit"]
        )
        spectral_feature = (
            discovery["fragment_mz_bin"]
            if selected_spectral_kind == "fragment_mz"
            else discovery["neutral_loss_bin"]
        )
        spectral_matrix = (
            peak_presence[confirmation_idx]
            if selected_spectral_kind == "fragment_mz"
            else loss_presence[confirmation_idx]
        )
        spectral_p = enrichment_p_value(spectral_matrix, confirmation_mask, spectral_feature)
        combined_p = float(combine_pvalues([structure_p, spectral_p], method="fisher").pvalue)

        mz_corr = safe_abs_corr(score, precursor)
        peak_corr = safe_abs_corr(score, peak_count)
        ce_corr = safe_abs_corr(score, collision_energy)
        instrument_eta2 = eta_squared(score, instrument)
        confounds = np.asarray([mz_corr, peak_corr, ce_corr, instrument_eta2], dtype=float)
        max_confound = float(np.nanmax(confounds)) if np.any(np.isfinite(confounds)) else float("nan")
        best_spectral = float(confirmed_spectral["log2_enrichment"])

        # Exploratory triage gate only.  It is intentionally conservative and is
        # not treated as a hypothesis test or proof of a chemical mechanism.
        candidate = bool(
            stability_mean[factor] >= 0.70
            and confirmed_structure["log2_enrichment"] >= 1.0
            and best_spectral >= 1.0
            and max_confound < 0.50
            and confirmed_structure["top_support"] >= 3
            and confirmed_spectral["top_support"] >= 3
        )
        row = {
            "factor": factor + 1,
            "stability_mean": stability_mean[factor],
            "stability_min": stability_min[factor],
            "explained_variance_before_rotation": pca.explained_variance_ratio_[factor],
            "abs_corr_precursor_mz": mz_corr,
            "abs_corr_peak_count": peak_corr,
            "abs_corr_collision_energy": ce_corr,
            "instrument_eta_squared": instrument_eta2,
            "max_confound_association": max_confound,
            "selected_pole": discovery["pole"],
            "morgan_bit": discovery["morgan_bit"],
            "representative_substructure": discovery["representative_substructure"],
            "annotation_morgan_log2_enrichment": discovery["morgan_log2_enrichment"],
            "best_morgan_log2_enrichment": confirmed_structure["log2_enrichment"],
            "morgan_top_support": confirmed_structure["top_support"],
            "morgan_top_prevalence": confirmed_structure["top_prevalence"],
            "morgan_background_prevalence": confirmed_structure["background_prevalence"],
            "fragment_mz_feature_index": discovery["fragment_mz_bin"],
            "fragment_mz_bin_Da": (
                (discovery["fragment_mz_bin"] + 0.5) * MZ_BIN_WIDTH
                if discovery["fragment_mz_bin"] >= 0 else float("nan")
            ),
            "annotation_fragment_log2_enrichment": discovery["fragment_log2_enrichment"],
            "fragment_log2_enrichment": confirmed_fragment["log2_enrichment"],
            "neutral_loss_feature_index": discovery["neutral_loss_bin"],
            "neutral_loss_bin_Da": (
                (discovery["neutral_loss_bin"] + 0.5) * LOSS_BIN_WIDTH
                if discovery["neutral_loss_bin"] >= 0 else float("nan")
            ),
            "annotation_neutral_loss_log2_enrichment": discovery["neutral_loss_log2_enrichment"],
            "neutral_loss_log2_enrichment": confirmed_loss["log2_enrichment"],
            "selected_spectral_kind": selected_spectral_kind,
            "selected_spectral_top_support": confirmed_spectral["top_support"],
            "selected_spectral_background_support": confirmed_spectral["background_support"],
            "best_spectral_log2_enrichment": best_spectral,
            "structure_enrichment_p": structure_p,
            "spectral_enrichment_p": spectral_p,
            "combined_enrichment_p": combined_p,
            "passes_effect_and_support_gate": candidate,
        }
        catalog_rows.append(row)

        ordered_local = confirmation_top_ids[
            np.argsort(np.abs(confirmation_score[confirmation_top_ids]))[::-1]
        ]
        for rank, local_idx in enumerate(ordered_local[:10], start=1):
            idx = int(confirmation_idx[int(local_idx)])
            item = holdout.iloc[idx]
            example_rows.append(
                {
                    "factor": factor + 1,
                    "pole": discovery["pole"],
                    "rank_within_pole": rank,
                    "factor_score": float(score[idx]),
                    "spectrum_id": item["spectrum_id"],
                    "inchikey_14": item["inchikey_14"],
                    "smiles": item["smiles"],
                    "precursor_mz": item["precursor_mz"],
                    "n_peaks": item["n_peaks"],
                    "collision_energy": item["ce"],
                    "instrument": item["instrument"],
                }
            )

    catalog = pd.DataFrame(catalog_rows)
    catalog["combined_enrichment_q"] = benjamini_hochberg(
        catalog["combined_enrichment_p"].to_numpy(float)
    )
    catalog["candidate_for_review"] = (
        catalog["passes_effect_and_support_gate"]
        & (catalog["combined_enrichment_q"] <= 0.10)
    )
    catalog["strictly_replicated_factor"] = (
        catalog["candidate_for_review"]
        & (catalog["structure_enrichment_p"] <= 0.05)
        & (catalog["spectral_enrichment_p"] <= 0.05)
    )
    examples = pd.DataFrame(example_rows)
    catalog.to_csv(args.output_dir / "factor_catalog.csv", index=False)
    examples.to_csv(args.output_dir / "factor_examples.csv", index=False)

    save_overview(catalog, pca.explained_variance_ratio_, args.output_dir / "embedding_factor_overview.png")
    save_molecule_grid(catalog, examples, args.output_dir / "top_factor_molecules.png")

    top = catalog.sort_values(
        ["candidate_for_review", "stability_mean", "best_morgan_log2_enrichment"],
        ascending=False,
    ).head(10)
    summary = {
        "protocol": (
            "PCA followed by orthogonal varimax rotation; no curated rule labels used; "
            "held-out molecules split again into annotation and independent confirmation halves"
        ),
        "all_spectra": len(manifest),
        "unique_molecules": len(selected),
        "train_molecules": len(train_idx),
        "heldout_molecules": len(holdout_idx),
        "annotation_molecules": len(annotation_idx),
        "confirmation_molecules": len(confirmation_idx),
        "n_factors": args.n_factors,
        "cumulative_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "candidate_factors_for_review": int(catalog["candidate_for_review"].sum()),
        "candidate_factor_ids": catalog.loc[catalog["candidate_for_review"], "factor"].astype(int).tolist(),
        "strictly_replicated_factors": int(catalog["strictly_replicated_factor"].sum()),
        "strictly_replicated_factor_ids": catalog.loc[
            catalog["strictly_replicated_factor"], "factor"
        ].astype(int).tolist(),
        "selection_gate": {
            "mean_resampling_loading_cosine_min": 0.70,
            "heldout_structure_log2_enrichment_min": 1.0,
            "heldout_peak_or_loss_log2_enrichment_min": 1.0,
            "largest_confound_association_max": 0.50,
            "combined_enrichment_BH_q_max": 0.10,
        },
        "important_limitation": (
            "A retained factor is a reproducible association, not a proven fragmentation mechanism. "
            "Peak occlusion/re-encoding and independent-dataset replication are still required."
        ),
    }
    (args.output_dir / "factor_discovery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Frozen DreaMS embedding：无规则潜在因子预实验\n",
        "## 目的\n",
        "在不使用 335/3,486 条规则标签的前提下，从冻结的 DreaMS precursor embedding 中发现可重复的潜在方向；随后仅在留出分子上检查这些方向是否同时富集局部结构片段与谱峰/中性丢失模式。\n",
        "## 协议\n",
        f"- 从 {len(manifest):,} 张谱图中，每个 IK14 只保留一张峰数最多的代表谱图，共 {len(selected):,} 个分子，避免多谱分子支配结果。",
        f"- 按分子划分：训练 {len(train_idx):,}，完全留出 {len(holdout_idx):,}；留出集再分为候选注释 {len(annotation_idx):,} 与独立确认 {len(confirmation_idx):,}。",
        f"- 训练端：{args.n_factors} 维 PCA + varimax 旋转；{args.stability_runs} 次子采样重拟合检查方向稳定性。",
        f"- 候选注释端：从 Morgan 原子环境、{MZ_BIN_WIDTH:.2f} Da 碎片 m/z bin 和 {LOSS_BIN_WIDTH:.2f} Da 中性丢失 bin 中选择一个候选含义。",
        "- 独立确认端：只检验已经选定的结构 bit 与谱图 bin，不重新挑选最大值。",
        "- 当前规则库未参与拟合、旋转、因子筛选或命名。\n",
        "## 结果\n",
        f"- 前 {args.n_factors} 个主成分累计解释方差：{np.sum(pca.explained_variance_ratio_):.3f}。",
        f"- 通过探索性复核门槛的因子：{int(catalog['candidate_for_review'].sum())}/{args.n_factors}。",
        f"- 结构和谱图证据分别达到单侧 p≤0.05 的严格复现因子：{int(catalog['strictly_replicated_factor'].sum())}/{args.n_factors}。",
        "",
        "| 因子 | 稳定性 | 结构富集(log2) | 谱峰/丢失富集(log2) | BH q | 最大混杂 | 代表结构片段 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in top.itertuples():
        report_lines.append(
            f"| F{int(row.factor):02d} | {row.stability_mean:.3f} | "
            f"{row.best_morgan_log2_enrichment:.2f} | {row.best_spectral_log2_enrichment:.2f} | "
            f"{row.combined_enrichment_q:.3g} | {row.max_confound_association:.2f} | "
            f"`{row.representative_substructure}` |"
        )
    report_lines += [
        "\n## 解释边界\n",
        "部分 embedding 方向本身在重采样后可复现，但当前没有一个方向同时通过独立结构复现、0.02 Da 谱图复现、混杂控制和多重检验校正。因此本轮没有可写入规则库的因子。下一轮应改用过完备稀疏自编码器，并同时分析 precursor embedding 与 peak-token embedding；只有候选重新出现后，才进入峰遮蔽重编码和独立数据集复现。",
    ]
    (args.output_dir / "FACTOR_DISCOVERY_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTop factors:\n", top[[
        "factor", "stability_mean", "best_morgan_log2_enrichment",
        "best_spectral_log2_enrichment", "max_confound_association",
        "combined_enrichment_q", "representative_substructure", "candidate_for_review"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
