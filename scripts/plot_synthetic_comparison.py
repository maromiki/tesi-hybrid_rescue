#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 180,
        "savefig.dpi": 300,
    }
)

MODELS = ["dmc", "4cac", "hybrid_no_rescue", "hybrid_rescue"]
MODEL_LABELS = {
    "dmc": "DeepMicroClass",
    "4cac": "4CAC",
    "hybrid_no_rescue": "Hybrid (no rescue)",
    "hybrid_rescue": "Hybrid (rescue)",
}
MODEL_COLORS = {
    "dmc": "#5DA5DA",
    "4cac": "#60BD68",
    "hybrid_no_rescue": "#fdae61",
    "hybrid_rescue": "#f46d43",
}

CLASSES = ["virus", "plasmid", "eukaryota", "bacteria"]
CLASS_TITLES = {
    "virus": "Classification of viruses",
    "plasmid": "Classification of plasmids",
    "eukaryota": "Classification of eukaryotes",
    "bacteria": "Classification of prokaryotes",
}


def _model_subset(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["model"].isin(MODELS)].copy()
    out["model"] = pd.Categorical(out["model"], categories=MODELS, ordered=True)
    out = out.sort_values("model")
    return out


def _save_pub_ready(fig: plt.Figure, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")


def _plot_grid_for_dataset(df_ds: pd.DataFrame, dataset_name: str, out_png: Path) -> None:
    data = _model_subset(df_ds)
    if data.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), constrained_layout=False)
    axes = axes.flatten()

    x_labels = ["Precision", "Recall", "F1 score"]
    x = np.arange(len(x_labels))
    width = 0.18

    for i, cls in enumerate(CLASSES):
        ax = axes[i]
        for j, model in enumerate(MODELS):
            row = data[data["model"] == model]
            if row.empty:
                vals = [0.0, 0.0, 0.0]
            else:
                vals = [
                    float(row[f"precision_{cls}"].iloc[0]),
                    float(row[f"recall_{cls}"].iloc[0]),
                    float(row[f"f1_{cls}"].iloc[0]),
                ]
            ax.bar(
                x + (j - (len(MODELS) - 1) / 2) * width,
                vals,
                width=width,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )

        ax.set_title(CLASS_TITLES[cls], pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3, linewidth=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"Synthetic datasets - {dataset_name}", fontsize=16, y=0.97)
    # Balanced spacing between panels and legend, avoiding overlap with x-axis labels.
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.14, wspace=0.20, hspace=0.30)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handlelength=1.6,
    )
    _save_pub_ready(fig, out_png)
    plt.close(fig)


def _plot_macro(df: pd.DataFrame, out_png: Path) -> None:
    data = _model_subset(df)
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=False)
    x_labels = ["Precision", "Recall", "F1 score"]
    x = np.arange(len(x_labels))
    width = 0.18

    for j, model in enumerate(MODELS):
        row = data[data["model"] == model]
        if row.empty:
            vals = [0.0, 0.0, 0.0]
        else:
            vals = [
                float(row["precision_macro"].iloc[0]),
                float(row["recall_macro"].iloc[0]),
                float(row["f1_macro"].iloc[0]),
            ]
        ax.bar(
            x + (j - (len(MODELS) - 1) / 2) * width,
            vals,
            width=width,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )

    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_title("Macro metrics", pad=10)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)
    _save_pub_ready(fig, out_png)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Create comparison plots for synthetic validation.")
    p.add_argument(
        "--metrics-pooled",
        default=".validazione/synthetic_metrics_pooled.tsv",
    )
    p.add_argument(
        "--out-dir",
        default=".validazione/plots",
    )
    args = p.parse_args()

    metrics = pd.read_csv(args.metrics_pooled, sep="\t")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep only pooled rows (sample=ALL), so missing samples are already excluded.
    pooled = metrics[metrics["sample"] == "ALL"].copy()
    pooled = pooled[pooled["model"].isin(MODELS)].copy()

    datasets: List[str] = [d for d in sorted(pooled["dataset"].unique()) if d != "ALL"]

    for ds in datasets:
        ds_df = pooled[pooled["dataset"] == ds].copy()
        _plot_grid_for_dataset(ds_df, ds, out_dir / f"comparison_{ds}.png")
        _plot_macro(ds_df, out_dir / f"comparison_{ds}_macro.png")

    all_df = pooled[pooled["dataset"] == "ALL"].copy()
    if not all_df.empty:
        _plot_grid_for_dataset(all_df, "ALL", out_dir / "comparison_all_synthetic.png")
        _plot_macro(all_df, out_dir / "comparison_all_synthetic_macro.png")

    pooled.to_csv(out_dir / "plot_input_pooled_used.tsv", sep="\t", index=False)
    print(f"Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
