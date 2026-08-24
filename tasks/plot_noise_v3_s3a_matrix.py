"""Create compact, publication-auditable visual summaries of Noise-v3 S3A."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/g8r_noise_v3_s3a_extended_matrix"),
    )
    return parser.parse_args()


def action_label(frame: pd.DataFrame) -> pd.Series:
    parts = frame["cell"].str.extract(
        r"^(?P<selector>.+)\|a=(?P<dose>[0-9.]+)\|step=(?P<step>[0-9]+)$"
    )
    if parts.isna().any().any():
        raise RuntimeError("malformed S3A cell name")
    frame["selector"] = parts["selector"]
    frame["dose"] = parts["dose"].astype(float)
    frame["step"] = parts["step"].astype(int)
    return frame["selector"] + "  a=" + frame["dose"].map(lambda x: f"{x:.2f}")


def heatmap(axis, frame: pd.DataFrame, value: str, title: str, cmap: str) -> None:
    table = frame.pivot(index="action", columns="step", values=value).sort_index()
    image = axis.imshow(table.to_numpy(float), aspect="auto", cmap=cmap)
    axis.set_title(title, fontsize=11, fontweight="bold")
    axis.set_xlabel("Sequential intervention step")
    axis.set_ylabel("")
    axis.set_xticks(range(len(table.columns)), table.columns)
    axis.set_yticks(range(len(table.index)), table.index, fontsize=8)
    values = table.to_numpy(float)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                axis.text(
                    column, row, f"{values[row, column]:.0f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(values[row, column]) > 0.55 * np.nanmax(abs(values)) else "black",
                )
    plt.colorbar(image, ax=axis, fraction=0.03, pad=0.02)


def main() -> None:
    args = parse_args()
    cell_path = args.output_dir / "cell_summary.csv"
    transition_path = args.output_dir / "transition_audit.csv.gz"
    if not cell_path.is_file() or not transition_path.is_file():
        raise FileNotFoundError("S3A decision outputs are incomplete")
    cells = pd.read_csv(cell_path)
    cells["action"] = action_label(cells)
    transitions = pd.read_csv(transition_path)

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    heatmap(axes[0, 0], cells, "corrected", "Official errors corrected", "Blues")
    heatmap(axes[0, 1], cells, "introduced", "New errors introduced", "Reds")
    heatmap(axes[1, 0], cells, "net", "Net corrections (corrected - introduced)", "RdYlGn")
    heatmap(
        axes[1, 1], cells, "unique_corrections_beyond_s1c_s2",
        "New error coverage beyond S1c + S2", "Purples",
    )
    figure.suptitle(
        "Noise-v3 S3A: preregistered error-mechanism action matrix",
        fontsize=15, fontweight="bold",
    )
    figure.savefig(args.output_dir / "s3a_action_matrix.png", dpi=220)
    plt.close(figure)

    changed = transitions.copy()
    changed["action"] = (
        changed["selector"].astype(str)
        + " a=" + changed["attenuation"].map(lambda x: f"{float(x):.2f}")
    )
    transition_counts = changed.groupby(
        ["action", "transition", "winner_mces_grade_name"], dropna=False,
    ).size().rename("queries").reset_index()
    transition_counts.to_csv(args.output_dir / "transition_grade_counts.csv", index=False)
    actions = sorted(changed["action"].unique())
    grades = ["near", "mid", "far", "unknown", "identity"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for axis, transition, title in zip(
        axes, ("corrected", "introduced"),
        ("Corrected errors: destination grade", "Introduced errors: wrong destination grade"),
    ):
        subset = transition_counts.loc[transition_counts["transition"] == transition]
        bottom = np.zeros(len(actions), dtype=float)
        for grade in grades:
            lookup = subset.loc[subset["winner_mces_grade_name"] == grade].set_index("action")["queries"]
            values = np.asarray([float(lookup.get(action, 0)) for action in actions])
            axis.barh(actions, values, left=bottom, label=grade)
            bottom += values
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Transition rows across registered steps")
        axis.grid(axis="x", alpha=0.2)
    axes[1].legend(title="MCES grade", loc="best")
    figure.savefig(args.output_dir / "s3a_transition_destinations.png", dpi=220)
    plt.close(figure)
    print(f"[visuals] {args.output_dir / 's3a_action_matrix.png'}", flush=True)
    print(f"[visuals] {args.output_dir / 's3a_transition_destinations.png'}", flush=True)


if __name__ == "__main__":
    main()
