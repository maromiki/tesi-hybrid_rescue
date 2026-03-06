#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / max(temperature, 1e-8)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def dmc_to_4class(
    dmc_tsv: Path,
    out_tsv: Path,
    temperature: float,
    anchor_threshold: float,
    fallback_probs_tsv: Path | None = None,
) -> None:
    df = pd.read_csv(dmc_tsv, sep="\t", low_memory=False)
    cols = ["Sequence Name", "Eukaryote", "EukaryoteVirus", "Plasmid", "Prokaryote", "ProkaryoteVirus"]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing DMC columns: {miss}")

    logits = df[["Eukaryote", "EukaryoteVirus", "Plasmid", "Prokaryote", "ProkaryoteVirus"]].to_numpy(dtype=float)

    # DeepMicroClass may emit all-zero rows for skipped/too-short contigs.
    # Treat these as low-confidence/unanchored instead of forcing a class via softmax.
    row_abs_sum = np.sum(np.abs(logits), axis=1)
    row_range = np.max(logits, axis=1) - np.min(logits, axis=1)
    invalid_rows = (row_abs_sum <= 1e-12) | (row_range <= 1e-12)

    probs5 = softmax(logits, temperature=temperature)

    # Use max viral component (not sum) to avoid systematic viral inflation.
    viral = np.maximum(probs5[:, 1], probs5[:, 4])
    plasmid = probs5[:, 2]
    bacteria = probs5[:, 3]
    euk = probs5[:, 0]

    out = pd.DataFrame(
        {
            "header": df["Sequence Name"].astype(str),
            "viral_score": viral,
            "plas_score": plasmid,
            "prokar_score": bacteria,
            "eukar_score": euk,
        }
    )
    conf = out[["viral_score", "plas_score", "prokar_score", "eukar_score"]].max(axis=1)
    uncertain = (conf < float(anchor_threshold)).to_numpy() | invalid_rows

    if fallback_probs_tsv is not None and fallback_probs_tsv.exists():
        fb = pd.read_csv(fallback_probs_tsv, sep="\t", usecols=["header", "viral_score", "plas_score", "prokar_score", "eukar_score"])
        out = out.merge(fb, on="header", how="left", suffixes=("", "_fb"))
        has_fb = out["viral_score_fb"].notna().to_numpy()
        use_fb = uncertain & has_fb
        for c in ["viral_score", "plas_score", "prokar_score", "eukar_score"]:
            out.loc[use_fb, c] = out.loc[use_fb, f"{c}_fb"]
        out = out[["header", "viral_score", "plas_score", "prokar_score", "eukar_score"]]
        uncertain = uncertain & (~has_fb)

    # Keep uncertain anchors below 0.1 so 4CAC does not auto-fallback to prokarya.
    out.loc[uncertain, ["viral_score", "plas_score", "prokar_score", "eukar_score"]] = 0.01
    out = out[["header", "viral_score", "plas_score", "prokar_score", "eukar_score"]]
    out.to_csv(out_tsv, sep="\t", index=False)


def run_4cac(
    fourcac_script: Path,
    conda_env: str,
    assembler: str,
    asmdir: Path,
    classdir: Path,
    outdir: Path,
    phage_thr: float,
    plasmid_thr: float,
) -> None:
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(fourcac_script),
        "--assembler",
        assembler,
        "--asmdir",
        f"{asmdir}/",
        "--classdir",
        f"{classdir}/",
        "--outdir",
        f"{outdir}/",
        "--phage",
        str(phage_thr),
        "--plasmid",
        str(plasmid_thr),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Run hybrid graph correction by feeding DMC softmax scores to 4CAC.")
    p.add_argument("--dmc", required=True)
    p.add_argument("--asmdir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--assembler", required=True, choices=["metaSPAdes", "metaFlye"])
    p.add_argument("--contig-base", required=True, help="Assembly FASTA basename (e.g. scaffolds.fasta or assembly.fasta)")
    p.add_argument("--fourcac-script", default="tools/4CAC/classify_4CAC.py")
    p.add_argument("--fourcac-env", default="4cac")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--anchor-threshold", type=float, default=0.60)
    p.add_argument("--fallback-probs", default=None, help="Optional 4CAC probs_xgb_4class.out used for low-confidence DMC rows")
    p.add_argument("--phage-threshold", type=float, default=0.7)
    p.add_argument("--plasmid-threshold", type=float, default=0.7)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probs_file = out_dir / f"{args.contig_base}.probs_xgb_4class.out"
    dmc_to_4class(
        Path(args.dmc),
        probs_file,
        temperature=args.temperature,
        anchor_threshold=args.anchor_threshold,
        fallback_probs_tsv=Path(args.fallback_probs) if args.fallback_probs else None,
    )

    run_4cac(
        fourcac_script=Path(args.fourcac_script),
        conda_env=args.fourcac_env,
        assembler=args.assembler,
        asmdir=Path(args.asmdir),
        classdir=out_dir,
        outdir=out_dir,
        phage_thr=args.phage_threshold,
        plasmid_thr=args.plasmid_threshold,
    )
    print(f"Hybrid output: {out_dir / '4CAC_classification.fasta'}")


if __name__ == "__main__":
    main()
