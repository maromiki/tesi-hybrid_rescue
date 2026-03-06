#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

CLASSES = ["Bacteria", "Eukaryota", "Plasmid", "Virus"]

DMC_MAP = {
    1: "Eukaryota",
    2: "Virus",
    3: "Plasmid",
    4: "Bacteria",
    5: "Virus",
}

CAC_MAP = {
    "prokarya": "Bacteria",
    "eukarya": "Eukaryota",
    "plasmid": "Plasmid",
    "phage": "Virus",
    "virus": "Virus",
    "uncertain": "Unknown",
}


def normalize_cac(x: object) -> str:
    if pd.isna(x):
        return "Unknown"
    return CAC_MAP.get(str(x).strip().lower(), "Unknown")


def load_dmc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", usecols=["Sequence Name", "best_choice"], low_memory=False)
    out = df.rename(columns={"Sequence Name": "contig_id"}).copy()
    out["pred_dmc"] = out["best_choice"].map(DMC_MAP).fillna("Unknown")
    return out[["contig_id", "pred_dmc"]]


def load_cac(path: Path, col_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["contig_id", "raw"], low_memory=False)
    df[col_name] = df["raw"].map(normalize_cac)
    return df[["contig_id", col_name]]


def evaluate_one(y_true: pd.Series, y_pred: pd.Series, mode: str, scenario: str, model: str) -> Dict[str, float]:
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, average="macro", zero_division=0
    )
    p_c, r_c, f1_c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, average=None, zero_division=0
    )
    row: Dict[str, float] = {
        "mode": mode,
        "scenario": scenario,
        "model": model,
        "n_eval": int(len(y_true)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
    }
    for i, c in enumerate(CLASSES):
        k = c.lower()
        row[f"precision_{k}"] = float(p_c[i])
        row[f"recall_{k}"] = float(r_c[i])
        row[f"f1_{k}"] = float(f1_c[i])
        row[f"support_{k}"] = int(sup[i])
    return row


def run_eval(
    mode: str,
    scenario_root: Path,
    dmc_file: Path,
    c4_file: Path,
    hybrid_file: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    dmc = load_dmc(dmc_file)
    c4 = load_cac(c4_file, "pred_4cac")
    hyb = load_cac(hybrid_file, "pred_hybrid_no_rescue")

    rows: List[Dict[str, float]] = []
    skip_rows: List[Dict[str, str]] = []

    for sc_dir in sorted([p for p in scenario_root.iterdir() if p.is_dir()]):
        scenario = sc_dir.name
        contigs_tsv = sc_dir / "contigs.tsv"
        if not contigs_tsv.exists():
            continue

        gt = pd.read_csv(contigs_tsv, sep="\t")
        gt = gt.rename(columns={"true_label": "true_label", "contig_id": "contig_id"})

        merged = gt[["contig_id", "true_label"]].merge(dmc, on="contig_id", how="inner")
        merged = merged.merge(c4, on="contig_id", how="inner")
        merged = merged.merge(hyb, on="contig_id", how="inner")

        if merged.empty:
            skip_rows.append({"mode": mode, "scenario": scenario, "reason": "empty_intersection"})
            continue

        merged["pred_hybrid_rescue"] = np.where(
            merged["pred_hybrid_no_rescue"].isin(["Bacteria", "Unknown"]) & (merged["pred_dmc"] == "Plasmid"),
            "Plasmid",
            merged["pred_hybrid_no_rescue"],
        )

        rows.append(evaluate_one(merged["true_label"], merged["pred_dmc"], mode, scenario, "dmc"))
        rows.append(evaluate_one(merged["true_label"], merged["pred_4cac"], mode, scenario, "4cac"))
        rows.append(
            evaluate_one(
                merged["true_label"],
                merged["pred_hybrid_no_rescue"],
                mode,
                scenario,
                "hybrid_no_rescue",
            )
        )
        rows.append(evaluate_one(merged["true_label"], merged["pred_hybrid_rescue"], mode, scenario, "hybrid_rescue"))

    metrics = pd.DataFrame(rows)
    skips = pd.DataFrame(skip_rows)

    metrics.to_csv(out_dir / f"metrics_{mode}_per_scenario.tsv", sep="\t", index=False)
    skips.to_csv(out_dir / f"skipped_{mode}.tsv", sep="\t", index=False)

    if not metrics.empty:
        compact = (
            metrics.groupby(["mode", "model"], as_index=False)[
                [
                    "precision_macro",
                    "recall_macro",
                    "f1_macro",
                    "precision_bacteria",
                    "recall_bacteria",
                    "f1_bacteria",
                    "precision_eukaryota",
                    "recall_eukaryota",
                    "f1_eukaryota",
                    "precision_plasmid",
                    "recall_plasmid",
                    "f1_plasmid",
                    "precision_virus",
                    "recall_virus",
                    "f1_virus",
                ]
            ]
            .mean()
        )
        compact.to_csv(out_dir / f"metrics_{mode}_mean.tsv", sep="\t", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate DMC/4CAC/hybrid on CAMISIM contig-level scenarios.")
    p.add_argument("--mode", required=True, choices=["short", "long"])
    p.add_argument("--scenario-root", required=True)
    p.add_argument("--dmc", required=True)
    p.add_argument("--c4", required=True)
    p.add_argument("--hybrid", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    run_eval(
        mode=args.mode,
        scenario_root=Path(args.scenario_root),
        dmc_file=Path(args.dmc),
        c4_file=Path(args.c4),
        hybrid_file=Path(args.hybrid),
        out_dir=Path(args.out_dir),
    )
    print(f"Evaluation complete for mode={args.mode}")


if __name__ == "__main__":
    main()
