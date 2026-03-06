#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def plot_one(metrics: pd.DataFrame, mode: str, scenario: str, out_file: Path) -> None:
    metrics = metrics[metrics["model"].isin(MODELS)].copy()
    metrics["model"] = pd.Categorical(metrics["model"], MODELS, ordered=True)
    metrics = metrics.sort_values("model")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=False)
    axes = axes.flatten()
    x = np.arange(3)
    width = 0.18

    for i, cls in enumerate(CLASSES):
        ax = axes[i]
        for j, m in enumerate(MODELS):
            row = metrics[metrics["model"] == m]
            vals = [0.0, 0.0, 0.0]
            if not row.empty:
                vals = [
                    float(row[f"precision_{cls}"].iloc[0]),
                    float(row[f"recall_{cls}"].iloc[0]),
                    float(row[f"f1_{cls}"].iloc[0]),
                ]
            ax.bar(x + (j - 1.5) * width, vals, width=width, color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        ax.set_xticks(x)
        ax.set_xticklabels(["Precision", "Recall", "F1 score"])
        ax.set_ylim(0, 1.02)
        ax.set_title(cls.capitalize())
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"CAMISIM {mode} - {scenario}", fontsize=16, y=0.97)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.14, wspace=0.2, hspace=0.28)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=4, frameon=False)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.suffix.lower() == ".png":
        fig.savefig(out_file, dpi=320, bbox_inches="tight")
    else:
        fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot class-wise results for CAMISIM contig scenarios.")
    p.add_argument("--metrics", required=True, help="metrics_<mode>_per_scenario.tsv")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", required=True, choices=["short", "long"])
    p.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Output formats, e.g. png pdf",
    )
    args = p.parse_args()

    df = pd.read_csv(args.metrics, sep="\t")
    out_dir = Path(args.out_dir)
    formats = [f.lower().lstrip(".") for f in args.formats]
    valid = {"png", "pdf", "svg"}
    bad = [f for f in formats if f not in valid]
    if bad:
        raise ValueError(f"Unsupported formats: {bad}. Allowed: {sorted(valid)}")

    for sc in sorted(df["scenario"].unique()):
        sc_df = df[df["scenario"] == sc].copy()
        for fmt in formats:
            plot_one(sc_df, args.mode, sc, out_dir / f"comparison_{args.mode}_{sc}.{fmt}")
    print(f"Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
