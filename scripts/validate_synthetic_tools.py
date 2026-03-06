#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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

GT_MAP = {
    "prokaryote": "Bacteria",
    "prokaryotic chromosomes": "Bacteria",
    "bacteria": "Bacteria",
    "eukaryote": "Eukaryota",
    "eukaryotic chromosomes": "Eukaryota",
    "eukaryota": "Eukaryota",
    "plasmid": "Plasmid",
    "prokaryotic plasmids": "Plasmid",
    "virus": "Virus",
    "prokaryotic virus": "Virus",
    "phage": "Virus",
    "uncertain": "Unknown",
    "unknown": "Unknown",
}


def normalize_text_label(x: object, mapping: Dict[str, str]) -> str:
    if pd.isna(x):
        return "Unknown"
    key = str(x).strip().lower()
    return mapping.get(key, "Unknown")


def load_gt(gt_file: Path) -> pd.DataFrame:
    df = pd.read_csv(gt_file, sep="\t", low_memory=False)

    if {"Contig_ID", "Ground_Truth_Class"}.issubset(df.columns):
        out = df[["Contig_ID", "Ground_Truth_Class"]].copy()
        out.columns = ["contig_id", "raw_label"]
    elif {"contig_id", "source_class"}.issubset(df.columns):
        out = df[["contig_id", "source_class"]].copy()
        out.columns = ["contig_id", "raw_label"]
    elif {"contig_id", "true_class"}.issubset(df.columns):
        out = df[["contig_id", "true_class"]].copy()
        out.columns = ["contig_id", "raw_label"]
    else:
        raise ValueError(f"Ground-truth format not recognized: {gt_file}")

    out["true_label"] = out["raw_label"].map(lambda x: normalize_text_label(x, GT_MAP))
    out = out[out["true_label"].isin(CLASSES)].copy()
    return out[["contig_id", "true_label"]]


def load_dmc(dmc_file: Path) -> pd.DataFrame:
    df = pd.read_csv(dmc_file, sep="\t", usecols=["Sequence Name", "best_choice"], low_memory=False)
    out = df.rename(columns={"Sequence Name": "contig_id"}).copy()
    out["pred_dmc"] = out["best_choice"].map(DMC_MAP).fillna("Unknown")
    return out[["contig_id", "pred_dmc"]]


def load_cac_like(pred_file: Path, is_hybrid_summary_tsv: bool) -> pd.DataFrame:
    if is_hybrid_summary_tsv:
        df = pd.read_csv(pred_file, sep="\t", usecols=["Contig_ID", "Classification"], low_memory=False)
        out = df.rename(columns={"Contig_ID": "contig_id", "Classification": "raw"}).copy()
    else:
        out = pd.read_csv(pred_file, header=None, names=["contig_id", "raw"], low_memory=False)

    out["pred"] = out["raw"].map(lambda x: normalize_text_label(x, CAC_MAP))
    return out[["contig_id", "pred"]]


def evaluate_model(y_true: pd.Series, y_pred: pd.Series, dataset: str, sample: str, model: str) -> Dict[str, float]:
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )
    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        average=None,
        zero_division=0,
    )

    out: Dict[str, float] = {
        "dataset": dataset,
        "sample": sample,
        "model": model,
        "n_eval": int(len(y_true)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
    }
    for i, c in enumerate(CLASSES):
        key = c.lower()
        out[f"precision_{key}"] = float(p_class[i])
        out[f"recall_{key}"] = float(r_class[i])
        out[f"f1_{key}"] = float(f1_class[i])
        out[f"support_{key}"] = int(support[i])
    return out


def find_first_tsv(folder: Path) -> Path | None:
    tsvs = sorted(folder.glob("*.tsv"))
    return tsvs[0] if tsvs else None


def build_jobs(root: Path) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []

    cami1_gt_dir = root / "output/cami_i_high/ground_truth_final_cami1"
    for gt in sorted(cami1_gt_dir.glob("*_GroundTruth.tsv")):
        sample = gt.stem.replace("_GroundTruth", "")
        dmc = find_first_tsv(root / "output/dmc/cami_i_high" / sample)
        c4 = root / "output/4cac/cami_i_high" / sample / "4CAC_classification.fasta"
        hyb = root / "output/hybrid/cami_i_high" / sample / "final_results/hybrid_classification_summary.tsv"
        jobs.append(
            {
                "dataset": "cami_i_high",
                "sample": sample,
                "gt": gt,
                "dmc": dmc,
                "c4": c4,
                "hyb": hyb,
                "hyb_is_summary_tsv": True,
            }
        )

    cami2_gt_dir = root / "output/cami_ii_marine/ground_truth_final"
    for gt in sorted(cami2_gt_dir.glob("*_GroundTruth.tsv")):
        sample = gt.stem.replace("_GroundTruth", "")
        dmc = find_first_tsv(root / "output/dmc/cami_ii_marine" / sample)
        c4 = root / "output/4cac/cami_ii_marine" / sample / "4CAC_classification.fasta"
        hyb = root / "output/hybrid/cami_ii_marine" / sample / "final_results/hybrid_classification_summary.tsv"
        jobs.append(
            {
                "dataset": "cami_ii_marine",
                "sample": sample,
                "gt": gt,
                "dmc": dmc,
                "c4": c4,
                "hyb": hyb,
                "hyb_is_summary_tsv": True,
            }
        )

    jobs.append(
        {
            "dataset": "camisim",
            "sample": "long_reads",
            "gt": root / "output/ground_truth/camisim/long_biased/assembly_gt_known.tsv",
            "dmc": root / "output/dmc/camisim/long_reads/contigs.fasta_pred_one-hot_hybrid.tsv",
            "c4": root / "output/4cac/camisim/long_reads/4CAC_classification.fasta",
            "hyb": root / "output/hybrid/camisim/long_reads/4CAC_classification.fasta",
            "hyb_is_summary_tsv": False,
        }
    )
    jobs.append(
        {
            "dataset": "camisim",
            "sample": "short_reads",
            "gt": root / "output/ground_truth/camisim/short_biased/scaffolds_gt_known.tsv",
            "dmc": root / "output/dmc/camisim/short_reads/scaffolds.fasta_pred_one-hot_hybrid.tsv",
            "c4": root / "output/4cac/camisim/short_reads/4CAC_classification.fasta",
            "hyb": root / "output/hybrid/camisim/short_reads/4CAC_classification.fasta",
            "hyb_is_summary_tsv": False,
        }
    )

    return jobs


def run_validation(root: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    skipped: List[Dict[str, str]] = []
    pooled_inputs: List[Dict[str, object]] = []

    for job in build_jobs(root):
        dataset = str(job["dataset"])
        sample = str(job["sample"])
        gt_file = Path(job["gt"])
        dmc_file = Path(job["dmc"]) if job["dmc"] is not None else None
        c4_file = Path(job["c4"])
        hyb_file = Path(job["hyb"])
        hyb_is_summary_tsv = bool(job["hyb_is_summary_tsv"])

        missing = []
        for name, fp in [
            ("gt", gt_file),
            ("dmc", dmc_file),
            ("4cac", c4_file),
            ("hybrid", hyb_file),
        ]:
            if fp is None or not fp.exists():
                missing.append(name)

        if missing:
            skipped.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "reason": f"missing_files:{','.join(missing)}",
                    "gt": str(gt_file),
                    "dmc": str(dmc_file) if dmc_file is not None else "",
                    "c4": str(c4_file),
                    "hybrid": str(hyb_file),
                }
            )
            continue

        gt = load_gt(gt_file)
        dmc = load_dmc(dmc_file)
        c4 = load_cac_like(c4_file, is_hybrid_summary_tsv=False).rename(columns={"pred": "pred_4cac"})
        hyb = load_cac_like(hyb_file, is_hybrid_summary_tsv=hyb_is_summary_tsv).rename(columns={"pred": "pred_hybrid_no_rescue"})

        merged = gt.merge(dmc, on="contig_id", how="inner")
        merged = merged.merge(c4, on="contig_id", how="inner")
        merged = merged.merge(hyb, on="contig_id", how="inner")

        if merged.empty:
            skipped.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "reason": "empty_intersection",
                    "gt": str(gt_file),
                    "dmc": str(dmc_file),
                    "c4": str(c4_file),
                    "hybrid": str(hyb_file),
                }
            )
            continue

        merged["pred_hybrid_rescue"] = np.where(
            merged["pred_hybrid_no_rescue"].isin(["Bacteria", "Unknown"]) & (merged["pred_dmc"] == "Plasmid"),
            "Plasmid",
            merged["pred_hybrid_no_rescue"],
        )

        rows.append(evaluate_model(merged["true_label"], merged["pred_dmc"], dataset, sample, "dmc"))
        rows.append(evaluate_model(merged["true_label"], merged["pred_4cac"], dataset, sample, "4cac"))
        rows.append(
            evaluate_model(
                merged["true_label"],
                merged["pred_hybrid_no_rescue"],
                dataset,
                sample,
                "hybrid_no_rescue",
            )
        )
        rows.append(evaluate_model(merged["true_label"], merged["pred_hybrid_rescue"], dataset, sample, "hybrid_rescue"))

        pooled_inputs.append(
            {
                "dataset": dataset,
                "sample": sample,
                "merged": merged[["true_label", "pred_dmc", "pred_4cac", "pred_hybrid_no_rescue", "pred_hybrid_rescue"]].copy(),
            }
        )

    per_sample = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)

    if per_sample.empty:
        per_sample.to_csv(out_dir / "synthetic_metrics_per_sample.tsv", sep="\t", index=False)
        skipped_df.to_csv(out_dir / "synthetic_skipped_samples.tsv", sep="\t", index=False)
        return per_sample, skipped_df

    mean_by_dataset = (
        per_sample.groupby(["dataset", "model"], as_index=False)[
            [
                "n_eval",
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

    pooled_rows: List[Dict[str, float]] = []

    for dataset in sorted({str(x["dataset"]) for x in pooled_inputs}):
        dfs = [x["merged"] for x in pooled_inputs if x["dataset"] == dataset]
        dcat = pd.concat(dfs, axis=0, ignore_index=True)
        pooled_rows.append(evaluate_model(dcat["true_label"], dcat["pred_dmc"], dataset, "ALL", "dmc"))
        pooled_rows.append(evaluate_model(dcat["true_label"], dcat["pred_4cac"], dataset, "ALL", "4cac"))
        pooled_rows.append(
            evaluate_model(dcat["true_label"], dcat["pred_hybrid_no_rescue"], dataset, "ALL", "hybrid_no_rescue")
        )
        pooled_rows.append(evaluate_model(dcat["true_label"], dcat["pred_hybrid_rescue"], dataset, "ALL", "hybrid_rescue"))

    all_cat = pd.concat([x["merged"] for x in pooled_inputs], axis=0, ignore_index=True)
    pooled_rows.append(evaluate_model(all_cat["true_label"], all_cat["pred_dmc"], "ALL", "ALL", "dmc"))
    pooled_rows.append(evaluate_model(all_cat["true_label"], all_cat["pred_4cac"], "ALL", "ALL", "4cac"))
    pooled_rows.append(
        evaluate_model(all_cat["true_label"], all_cat["pred_hybrid_no_rescue"], "ALL", "ALL", "hybrid_no_rescue")
    )
    pooled_rows.append(evaluate_model(all_cat["true_label"], all_cat["pred_hybrid_rescue"], "ALL", "ALL", "hybrid_rescue"))
    pooled_df = pd.DataFrame(pooled_rows)

    per_sample.to_csv(out_dir / "synthetic_metrics_per_sample.tsv", sep="\t", index=False)
    mean_by_dataset.to_csv(out_dir / "synthetic_metrics_mean_by_dataset.tsv", sep="\t", index=False)
    pooled_df.to_csv(out_dir / "synthetic_metrics_pooled.tsv", sep="\t", index=False)
    skipped_df.to_csv(out_dir / "synthetic_skipped_samples.tsv", sep="\t", index=False)

    return per_sample, skipped_df


def main() -> None:
    p = argparse.ArgumentParser(description="Validation DMC vs 4CAC vs Hybrid on synthetic datasets.")
    p.add_argument(
        "--root",
        default="/nfsd/bcb/bcbg/marongiumi",
        help="Workspace root that contains output/* folders.",
    )
    p.add_argument(
        "--out-dir",
        default="/nfsd/bcb/bcbg/marongiumi/thesis/finalissima/validazione",
        help="Output folder for validation results.",
    )
    args = p.parse_args()

    per_sample, skipped = run_validation(Path(args.root), Path(args.out_dir))

    if per_sample.empty:
        print("No valid sample could be evaluated.")
        return

    print("Validation completed.")
    print(f"Rows written (per-sample): {len(per_sample)}")
    print(f"Skipped samples: {len(skipped)}")
    summary = (
        per_sample.groupby(["dataset", "model"], as_index=False)[["precision_macro", "recall_macro", "f1_macro"]]
        .mean()
        .sort_values(["dataset", "f1_macro"], ascending=[True, False])
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
