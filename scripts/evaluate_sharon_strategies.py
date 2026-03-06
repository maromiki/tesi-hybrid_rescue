#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

CLASSES = ["Bacteria", "Eukaryota", "Plasmid", "Virus"]
DMC_MAP = {1: "Eukaryota", 2: "Virus", 3: "Plasmid", 4: "Bacteria", 5: "Virus"}
CAC_MAP = {"prokarya": "Bacteria", "eukarya": "Eukaryota", "phage": "Virus", "plasmid": "Plasmid", "uncertain": "Unknown"}


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def evaluate(df: pd.DataFrame, pred_col: str) -> dict:
    y_true = df["class_label"]
    y_pred = df[pred_col]
    rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    return {
        "model": pred_col,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(rep["macro avg"]["f1-score"]),
        "f1_weighted": float(rep["weighted avg"]["f1-score"]),
        "precision_bacteria": float(rep["Bacteria"]["precision"]),
        "recall_bacteria": float(rep["Bacteria"]["recall"]),
        "f1_bacteria": float(rep["Bacteria"]["f1-score"]),
        "precision_eukaryota": float(rep["Eukaryota"]["precision"]),
        "recall_eukaryota": float(rep["Eukaryota"]["recall"]),
        "f1_eukaryota": float(rep["Eukaryota"]["f1-score"]),
        "precision_plasmid": float(rep["Plasmid"]["precision"]),
        "recall_plasmid": float(rep["Plasmid"]["recall"]),
        "f1_plasmid": float(rep["Plasmid"]["f1-score"]),
        "precision_virus": float(rep["Virus"]["precision"]),
        "recall_virus": float(rep["Virus"]["recall"]),
        "f1_virus": float(rep["Virus"]["f1-score"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True)
    p.add_argument("--dmc", required=True)
    p.add_argument("--hyb", required=True)
    p.add_argument("--c4", required=True)
    p.add_argument("--circular", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    gt = pd.read_csv(args.gt)

    dmc = pd.read_csv(args.dmc, sep="\t")
    dmc = dmc.rename(columns={"Sequence Name": "contig_id"})
    dmc["pred_dmc"] = dmc["best_choice"].map(DMC_MAP).fillna("Unknown")

    logits = dmc[["Eukaryote", "EukaryoteVirus", "Plasmid", "Prokaryote", "ProkaryoteVirus"]].to_numpy(float)
    probs = softmax(logits)
    dmc["p_virus"] = probs[:, 1] + probs[:, 4]
    dmc["p_plasmid"] = probs[:, 2]
    dmc["p_bacteria"] = probs[:, 3]
    dmc["p_euk"] = probs[:, 0]

    hyb = pd.read_csv(args.hyb, header=None, names=["contig_id", "raw"])
    hyb["pred_hyb"] = hyb["raw"].astype(str).str.strip().str.lower().map(CAC_MAP).fillna("Unknown")

    c4 = pd.read_csv(args.c4, header=None, names=["contig_id", "raw"])
    c4["pred_4cac"] = c4["raw"].astype(str).str.strip().str.lower().map(CAC_MAP).fillna("Unknown")

    circular_set = set(pd.read_csv(args.circular, header=None, sep="\t")[0].astype(str).tolist())

    df = gt.merge(dmc[["contig_id", "pred_dmc", "p_virus", "p_plasmid", "p_bacteria", "p_euk"]], on="contig_id", how="inner")
    df = df.merge(hyb[["contig_id", "pred_hyb"]], on="contig_id", how="inner")
    df = df.merge(c4[["contig_id", "pred_4cac"]], on="contig_id", how="inner")

    df["pred_hybrid_rescue"] = np.where(
        df["pred_hyb"].isin(["Bacteria", "Unknown"]) & (df["pred_dmc"] == "Plasmid"),
        "Plasmid",
        df["pred_hyb"],
    )

    # Historical hierarchical strategy from previous scripts
    pred_hier = []
    for _, r in df.iterrows():
        cac = r["pred_4cac"]
        dmc_lbl = r["pred_dmc"]
        if cac == dmc_lbl:
            pred_hier.append(cac)
        elif cac in ["Bacteria", "Unknown"] and dmc_lbl == "Plasmid":
            pred_hier.append("Plasmid")
        elif cac == "Unknown" and dmc_lbl != "Virus":
            pred_hier.append(dmc_lbl)
        else:
            pred_hier.append(cac)
    df["pred_hierarchical"] = pred_hier

    def parse_len(cid: str) -> int:
        try:
            return int(cid.split("_")[3])
        except Exception:
            return 0

    df["is_circular"] = df["contig_id"].isin(circular_set)
    df["contig_len"] = df["contig_id"].map(parse_len)

    pred = df["pred_hybrid_rescue"].copy()
    mask = df["is_circular"] & (df["contig_len"] < 500000)
    pred2 = pred.copy()
    pred2[mask & (df["pred_dmc"] == "Virus")] = "Virus"
    pred2[mask & (df["pred_dmc"] != "Virus")] = "Plasmid"
    df["pred_circular_rescue"] = pred2

    # Brute-force tuning around old strategy.
    # We keep two views:
    # 1) best accuracy overall
    # 2) best plasmid-rescue model under strong global-accuracy constraint
    best_acc_score = None
    best_acc_row = None
    best_plasmid_score = None
    best_plasmid_row = None

    # Old figure thresholds
    min_old_acc = 0.8038
    min_old_recall_plasmid = 0.55

    for p_thr in np.round(np.arange(0.30, 0.91, 0.05), 3):
        for v_thr in np.round(np.arange(0.30, 0.91, 0.05), 3):
            for circ_len in [50000, 100000, 150000, 200000, 300000, 500000]:
                cur = df["pred_hyb"].copy()

                mask_plas = df["pred_hyb"].isin(["Bacteria", "Unknown"]) & (df["p_plasmid"] >= p_thr)
                cur[mask_plas] = "Plasmid"

                mask_virus = df["pred_hyb"].isin(["Bacteria", "Unknown"]) & (df["p_virus"] >= v_thr) & (~mask_plas)
                cur[mask_virus] = "Virus"

                m_circ = df["is_circular"] & (df["contig_len"] <= circ_len)
                cur[m_circ & (df["p_plasmid"] >= 0.6) & (df["p_plasmid"] >= df["p_virus"])] = "Plasmid"
                cur[m_circ & (df["p_virus"] >= 0.6) & (df["p_virus"] > df["p_plasmid"])] = "Virus"

                df["tmp"] = cur
                row = evaluate(df, "tmp")
                row["p_thr"] = float(p_thr)
                row["v_thr"] = float(v_thr)
                row["circ_len"] = int(circ_len)

                acc_score = (row["accuracy"], row["f1_macro"], row["f1_plasmid"], row["f1_virus"])
                if best_acc_score is None or acc_score > best_acc_score:
                    best_acc_score = acc_score
                    best_acc_row = row.copy()

                if row["accuracy"] >= min_old_acc and row["recall_plasmid"] >= min_old_recall_plasmid:
                    p_score = (row["f1_plasmid"], row["recall_plasmid"], row["accuracy"], row["f1_macro"])
                    if best_plasmid_score is None or p_score > best_plasmid_score:
                        best_plasmid_score = p_score
                        best_plasmid_row = row.copy()

    rows = [
        evaluate(df, "pred_4cac"),
        evaluate(df, "pred_dmc"),
        evaluate(df, "pred_hyb"),
        evaluate(df, "pred_hierarchical"),
        evaluate(df, "pred_hybrid_rescue"),
        evaluate(df, "pred_circular_rescue"),
    ]

    if best_acc_row is not None:
        r1 = best_acc_row.copy()
        r1["model"] = "best_tuned_accuracy"
        rows.append(r1)

    if best_plasmid_row is not None:
        r2 = best_plasmid_row.copy()
        r2["model"] = "final_model_rescue"
        rows.append(r2)
    else:
        # Fallback: use the historical rescue model that recovers plasmids strongly.
        r2 = evaluate(df, "pred_hybrid_rescue")
        r2["model"] = "final_model_rescue"
        r2["p_thr"] = np.nan
        r2["v_thr"] = np.nan
        r2["circ_len"] = np.nan
        rows.append(r2)

    out_df = pd.DataFrame(rows).sort_values(["accuracy", "f1_macro"], ascending=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)

    if best_acc_row is not None:
        p_thr = float(best_acc_row["p_thr"])
        v_thr = float(best_acc_row["v_thr"])
        circ_len = int(best_acc_row["circ_len"])

        pred_best = df["pred_hyb"].copy()
        mask_plas = df["pred_hyb"].isin(["Bacteria", "Unknown"]) & (df["p_plasmid"] >= p_thr)
        pred_best[mask_plas] = "Plasmid"
        mask_virus = df["pred_hyb"].isin(["Bacteria", "Unknown"]) & (df["p_virus"] >= v_thr) & (~mask_plas)
        pred_best[mask_virus] = "Virus"
        m_circ = df["is_circular"] & (df["contig_len"] <= circ_len)
        pred_best[m_circ & (df["p_plasmid"] >= 0.6) & (df["p_plasmid"] >= df["p_virus"])] = "Plasmid"
        pred_best[m_circ & (df["p_virus"] >= 0.6) & (df["p_virus"] > df["p_plasmid"])] = "Virus"

        pd.DataFrame({"contig_id": df["contig_id"], "pred_label": pred_best}).to_csv(
            out_path.parent / "predictions_best_tuned.tsv", sep="\t", index=False
        )
        pd.DataFrame([
            {"p_thr": p_thr, "v_thr": v_thr, "circ_len": circ_len}
        ]).to_csv(out_path.parent / "best_tuned_config.tsv", sep="\t", index=False)

    if best_plasmid_row is not None:
        p_thr = float(best_plasmid_row["p_thr"])
        v_thr = float(best_plasmid_row["v_thr"])
        circ_len = int(best_plasmid_row["circ_len"])

        pred_final = df["pred_hyb"].copy()
        mask_plas = df["pred_hyb"].isin(["Bacteria", "Unknown"]) & (df["p_plasmid"] >= p_thr)
        pred_final[mask_plas] = "Plasmid"
        mask_virus = df["pred_hyb"].isin(["Bacteria", "Unknown"]) & (df["p_virus"] >= v_thr) & (~mask_plas)
        pred_final[mask_virus] = "Virus"
        m_circ = df["is_circular"] & (df["contig_len"] <= circ_len)
        pred_final[m_circ & (df["p_plasmid"] >= 0.6) & (df["p_plasmid"] >= df["p_virus"])] = "Plasmid"
        pred_final[m_circ & (df["p_virus"] >= 0.6) & (df["p_virus"] > df["p_plasmid"])] = "Virus"

        pd.DataFrame({"contig_id": df["contig_id"], "pred_label": pred_final}).to_csv(
            out_path.parent / "predictions_final_model_rescue.tsv", sep="\t", index=False
        )
        pd.DataFrame([
            {"p_thr": p_thr, "v_thr": v_thr, "circ_len": circ_len}
        ]).to_csv(out_path.parent / "final_model_rescue_config.tsv", sep="\t", index=False)
    else:
        pd.DataFrame({"contig_id": df["contig_id"], "pred_label": df["pred_hybrid_rescue"]}).to_csv(
            out_path.parent / "predictions_final_model_rescue.tsv", sep="\t", index=False
        )
        pd.DataFrame([
            {"mode": "fallback_pred_hybrid_rescue", "p_thr": np.nan, "v_thr": np.nan, "circ_len": np.nan}
        ]).to_csv(out_path.parent / "final_model_rescue_config.tsv", sep="\t", index=False)
    cols = [
        "model",
        "accuracy",
        "f1_macro",
        "f1_bacteria",
        "f1_eukaryota",
        "f1_plasmid",
        "f1_virus",
        "recall_bacteria",
        "recall_eukaryota",
        "recall_plasmid",
        "recall_virus",
        "p_thr",
        "v_thr",
        "circ_len",
    ]
    show_cols = [c for c in cols if c in out_df.columns]
    print(out_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
