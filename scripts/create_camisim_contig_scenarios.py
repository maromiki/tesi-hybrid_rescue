#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

CLASSES = ["Bacteria", "Eukaryota", "Plasmid", "Virus"]

GT_MAP = {
    "prokaryotic chromosomes": "Bacteria",
    "prokaryote": "Bacteria",
    "bacteria": "Bacteria",
    "eukaryotic chromosomes": "Eukaryota",
    "eukaryote": "Eukaryota",
    "eukaryota": "Eukaryota",
    "prokaryotic plasmids": "Plasmid",
    "plasmid": "Plasmid",
    "prokaryotic virus": "Virus",
    "virus": "Virus",
    "phage": "Virus",
}


def normalize_gt_label(row: pd.Series) -> str:
    """Map one ground-truth row to a normalized 4-class label.

    Args:
        row: Input row containing one or more class columns such as
            `source_class`, `true_class`, or `Ground_Truth_Class`.

    Returns:
        Normalized label in {"Bacteria", "Eukaryota", "Plasmid", "Virus"}
        when recognized, otherwise "Unknown".
    """
    for col in ("source_class", "true_class", "Ground_Truth_Class"):
        if col in row and pd.notna(row[col]):
            key = str(row[col]).strip().lower()
            if key in GT_MAP:
                return GT_MAP[key]
    return "Unknown"


def read_fasta_lengths(fasta_path: Path) -> Dict[str, int]:
    """Read sequence lengths from a FASTA file.

    Args:
        fasta_path: Path to the input FASTA file.

    Returns:
        Dictionary mapping contig IDs to sequence length in base pairs.
    """
    lengths: Dict[str, int] = {}
    cid = None
    clen = 0
    with fasta_path.open("rt") as f:
        for line in f:
            if line.startswith(">"):
                if cid is not None:
                    lengths[cid] = clen
                cid = line[1:].split()[0]
                clen = 0
            else:
                clen += len(line.strip())
    if cid is not None:
        lengths[cid] = clen
    return lengths


def write_subset_fasta(source_fasta: Path, keep_ids: set[str], out_fasta: Path) -> None:
    """Write only selected sequence records to a new FASTA file.

    Args:
        source_fasta: Source FASTA path containing all contigs.
        keep_ids: Set of contig IDs to retain.
        out_fasta: Output FASTA path for the subset.

    Returns:
        None. The subset FASTA is written to disk.
    """
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    with source_fasta.open("rt") as fin, out_fasta.open("wt") as fout:
        write = False
        for line in fin:
            if line.startswith(">"):
                cid = line[1:].split()[0]
                write = cid in keep_ids
            if write:
                fout.write(line)


def round_counts(total: int, fractions: Dict[str, float]) -> Dict[str, int]:
    """Convert class fractions into integer class counts.

    Args:
        total: Total number of contigs to allocate.
        fractions: Class-to-fraction mapping for the 4 target classes.

    Returns:
        Dictionary of integer class counts that sums exactly to `total`.
    """
    raw = {c: total * float(fractions[c]) for c in CLASSES}
    base = {c: int(np.floor(raw[c])) for c in CLASSES}
    remainder = total - sum(base.values())
    if remainder > 0:
        ordered = sorted(CLASSES, key=lambda c: (raw[c] - base[c]), reverse=True)
        for i in range(remainder):
            base[ordered[i % len(ordered)]] += 1
    return base


def create_scenarios(
    gt_file: Path,
    assembly_fasta: Path,
    config_json: Path,
    out_dir: Path,
    seed: int,
    min_length: int,
) -> None:
    """Generate contig subset scenarios from a full assembly and ground truth.

    Args:
        gt_file: Ground-truth TSV path.
        assembly_fasta: Assembly FASTA path used for sequence extraction.
        config_json: JSON configuration with scenario fractions.
        out_dir: Output directory where scenario folders are created.
        seed: Random seed for reproducible sampling.
        min_length: Minimum contig length (bp) to include.

    Returns:
        None. Scenario tables and FASTA subsets are written to disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(gt_file, sep="\t", low_memory=False)
    if "contig_id" not in gt.columns:
        if "Contig_ID" in gt.columns:
            gt = gt.rename(columns={"Contig_ID": "contig_id"})
        else:
            raise ValueError("GT file must contain 'contig_id' or 'Contig_ID'.")

    gt["true_label"] = gt.apply(normalize_gt_label, axis=1)
    gt = gt[gt["true_label"].isin(CLASSES)].copy()

    lengths = read_fasta_lengths(assembly_fasta)
    gt["length_bp"] = gt["contig_id"].map(lengths).fillna(0).astype(int)
    gt = gt[gt["length_bp"] >= min_length].copy()
    gt = gt.drop_duplicates(subset=["contig_id"], keep="first").copy()

    cfg = json.loads(config_json.read_text())
    total_contigs = int(cfg["total_contigs"])
    scenarios: List[Dict[str, object]] = cfg["scenarios"]

    rng = np.random.default_rng(seed)

    class_pools = {
        c: gt.loc[gt["true_label"] == c, "contig_id"].tolist() for c in CLASSES
    }

    summary_rows: List[Dict[str, object]] = []

    for sc in scenarios:
        name = str(sc["name"])
        fractions = sc["fractions"]
        req = round_counts(total_contigs, fractions)

        chosen: List[str] = []
        per_class_rows: List[Dict[str, object]] = []

        for c in CLASSES:
            need = int(req[c])
            pool = class_pools[c]
            if need > len(pool):
                raise ValueError(
                    f"Scenario '{name}' impossible without replacement for class {c}: "
                    f"requested {need}, available {len(pool)}"
                )
            pick = rng.choice(pool, size=need, replace=False).tolist()
            chosen.extend(pick)
            per_class_rows.append(
                {
                    "scenario": name,
                    "class": c,
                    "requested": need,
                    "selected": len(pick),
                    "available": len(pool),
                }
            )

        sel = gt[gt["contig_id"].isin(chosen)].copy()
        sc_dir = out_dir / name
        sc_dir.mkdir(parents=True, exist_ok=True)

        sel[["contig_id", "true_label", "length_bp"]].sort_values("contig_id").to_csv(
            sc_dir / "contigs.tsv", sep="\t", index=False
        )
        write_subset_fasta(assembly_fasta, set(chosen), sc_dir / "contigs.fasta")
        pd.DataFrame(per_class_rows).to_csv(sc_dir / "class_allocation.tsv", sep="\t", index=False)

        obs = sel["true_label"].value_counts().to_dict()
        summary_rows.append(
            {
                "scenario": name,
                "n_total": int(len(sel)),
                **{f"n_{c.lower()}": int(obs.get(c, 0)) for c in CLASSES},
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "scenario_summary.tsv", sep="\t", index=False)
    gt[["contig_id", "true_label", "length_bp"]].to_csv(out_dir / "contig_class_map_full.tsv", sep="\t", index=False)


def main() -> None:
    """Parse CLI arguments and run scenario generation."""
    p = argparse.ArgumentParser(description="Create 1000-contig CAMISIM scenario datasets from assembled contigs.")
    p.add_argument("--gt", required=True, help="Ground-truth TSV with contig_id and class columns")
    p.add_argument("--assembly", required=True, help="Assembly FASTA used to compute lengths and export scenario FASTA")
    p.add_argument("--config", required=True, help="Scenario JSON config")
    p.add_argument("--out-dir", required=True, help="Output directory for scenarios")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-length", type=int, default=500)
    args = p.parse_args()

    create_scenarios(
        gt_file=Path(args.gt),
        assembly_fasta=Path(args.assembly),
        config_json=Path(args.config),
        out_dir=Path(args.out_dir),
        seed=args.seed,
        min_length=args.min_length,
    )
    print(f"Scenario datasets created in: {args.out_dir}")


if __name__ == "__main__":
    main()
