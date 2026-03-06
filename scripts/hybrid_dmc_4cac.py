#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support

CLASS_ORDER = ["Virus", "Plasmid", "Bacteria", "Eukaryota"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
FOURCAC_NAME_MAP = {
    "phage": "Virus",
    "virus": "Virus",
    "plasmid": "Plasmid",
    "prokarya": "Bacteria",
    "eukarya": "Eukaryota",
    "uncertain": "Unknown",
}


@dataclass
class GraphData:
    node_index: Dict[int, int]
    node_ids: List[int]
    node_lengths: np.ndarray
    adjacency: List[List[int]]


@dataclass
class ContigPaths:
    contig_to_index: Dict[str, int]
    contig_names: List[str]
    contig_lengths: np.ndarray
    contig_nodes: List[List[int]]



def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / max(temperature, 1e-8)
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def load_dmc_probabilities(dmc_tsv: Path, temperature: float) -> pd.DataFrame:
    df = pd.read_csv(dmc_tsv, sep="\t")
    needed = ["Sequence Name", "Eukaryote", "EukaryoteVirus", "Plasmid", "Prokaryote", "ProkaryoteVirus"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing DMC columns: {missing}")

    logits = df[["Eukaryote", "EukaryoteVirus", "Plasmid", "Prokaryote", "ProkaryoteVirus"]].to_numpy(dtype=np.float64)
    probs5 = _softmax(logits, temperature=temperature)

    viral = probs5[:, 1] + probs5[:, 4]
    plasmid = probs5[:, 2]
    #!/usr/bin/env python3
    import argparse
    import json
    import subprocess
    from pathlib import Path
    from typing import Dict, List

    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

    CLASS_ORDER = ["Virus", "Plasmid", "Bacteria", "Eukaryota"]
    FOURCAC_NAME_MAP = {
        "phage": "Virus",
        "virus": "Virus",
        "plasmid": "Plasmid",
        "prokarya": "Bacteria",
        "eukarya": "Eukaryota",
        "uncertain": "Unknown",
    }


    def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
        z = logits / max(temperature, 1e-8)
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)


    def load_dmc_probabilities(dmc_tsv: Path, temperature: float) -> pd.DataFrame:
        df = pd.read_csv(dmc_tsv, sep="\t")
        cols = ["Sequence Name", "Eukaryote", "EukaryoteVirus", "Plasmid", "Prokaryote", "ProkaryoteVirus"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing DMC columns: {missing}")

        logits = df[["Eukaryote", "EukaryoteVirus", "Plasmid", "Prokaryote", "ProkaryoteVirus"]].to_numpy(dtype=np.float64)
        probs5 = _softmax(logits, temperature)

        viral = probs5[:, 1] + probs5[:, 4]
        plasmid = probs5[:, 2]
        bacteria = probs5[:, 3]
        euk = probs5[:, 0]
        probs4 = np.stack([viral, plasmid, bacteria, euk], axis=1)
        probs4 = probs4 / probs4.sum(axis=1, keepdims=True)

        out = pd.DataFrame(
            {
                "contig_id": df["Sequence Name"].astype(str),
                "viral_score": probs4[:, 0],
                "plas_score": probs4[:, 1],
                "prokar_score": probs4[:, 2],
                "eukar_score": probs4[:, 3],
            }
        )
        out["confidence"] = out[["viral_score", "plas_score", "prokar_score", "eukar_score"]].max(axis=1)
        out["pred_label_dmc"] = out[["viral_score", "plas_score", "prokar_score", "eukar_score"]].idxmax(axis=1).map(
            {
                "viral_score": "Virus",
                "plas_score": "Plasmid",
                "prokar_score": "Bacteria",
                "eukar_score": "Eukaryota",
            }
        )
        return out


    def read_spades_paths(paths_file: Path) -> pd.DataFrame:
        contigs = []
        node_list = []
        read_nodes = False
        cur_name = None
        cur_len = None

        with paths_file.open() as f:
            for raw in f:
                row = raw.rstrip("\n")
                if row.startswith("NODE"):
                    if row.endswith("'"):
                        read_nodes = False
                        continue
                    if cur_name is not None:
                        contigs.append((cur_name, cur_len, node_list[:]))
                    read_nodes = True
                    node_list = []
                    cur_name = row
                    cur_len = int(row.split("_")[3])
                else:
                    if not read_nodes:
                        continue
                    if row.endswith(";"):
                        row = row[:-1]
                    if not row:
                        continue
                    for x in row.split(","):
                        node_list.append(int(x[:-1]))

        if cur_name is not None:
            contigs.append((cur_name, cur_len, node_list[:]))

        return pd.DataFrame(contigs, columns=["contig_id", "length", "nodes"])


    def read_gfa_adjacency(gfa_file: Path) -> Dict[int, set]:
        adj: Dict[int, set] = {}
        with gfa_file.open() as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if parts[0] == "S":
                    nid = int(parts[1])
                    adj.setdefault(nid, set())
                elif parts[0] == "L":
                    a = int(parts[1])
                    b = int(parts[3])
                    if a == b:
                        continue
                    adj.setdefault(a, set()).add(b)
                    adj.setdefault(b, set()).add(a)
        return adj


    def is_isolated_contig(nodes: List[int], adj: Dict[int, set]) -> bool:
        own = set(nodes)
        if not own:
            return False
        for n in own:
            for m in adj.get(n, set()):
                if m not in own:
                    return False
        return True


    def write_4cac_input(dmc_probs: pd.DataFrame, out_file: Path, anchor_threshold: float) -> Dict[str, float]:
        x = dmc_probs.copy()
        uncertain = x["confidence"] < anchor_threshold
        # Keep original DMC probability vectors also for uncertain contigs.
        # The anchor threshold is retained for reporting/diagnostic purposes.
        x = x.rename(columns={"contig_id": "header"})
        x[["header", "viral_score", "plas_score", "prokar_score", "eukar_score"]].to_csv(out_file, sep="\t", index=False)
        return {
            "anchors_n": float((~uncertain).sum()),
            "anchors_rate": float((~uncertain).mean()),
        }


    def run_4cac(
        fourcac_script: Path,
        fourcac_env: str,
        asmdir: Path,
        classdir: Path,
        outdir: Path,
        phage_threshold: float,
        plasmid_threshold: float,
    ) -> None:
        asmdir_s = str(asmdir) if str(asmdir).endswith("/") else f"{asmdir}/"
        classdir_s = str(classdir) if str(classdir).endswith("/") else f"{classdir}/"
        outdir_s = str(outdir) if str(outdir).endswith("/") else f"{outdir}/"
        cmd = [
            "conda",
            "run",
            "-n",
            fourcac_env,
            "python",
            str(fourcac_script),
            "--assembler",
            "metaSPAdes",
            "--asmdir",
            asmdir_s,
            "--classdir",
            classdir_s,
            "--outdir",
            outdir_s,
            "--phage",
            str(phage_threshold),
            "--plasmid",
            str(plasmid_threshold),
        ]
        subprocess.run(cmd, check=True)


    def load_4cac_output(c4_file: Path) -> pd.DataFrame:
        df = pd.read_csv(c4_file, header=None, names=["contig_id", "raw"])
        df["raw"] = df["raw"].astype(str).str.strip().str.lower()
        df["pred_label"] = df["raw"].map(FOURCAC_NAME_MAP).fillna("Unknown")
        return df[["contig_id", "pred_label"]]


    def apply_plasmid_rescue(
        pred_df: pd.DataFrame,
        dmc_probs: pd.DataFrame,
        contig_nodes_df: pd.DataFrame,
        adjacency: Dict[int, set],
        plasmid_rescue_threshold: float,
    ) -> pd.DataFrame:
        iso = {
            row.contig_id: is_isolated_contig(row.nodes, adjacency)
            for row in contig_nodes_df.itertuples(index=False)
        }
        dmc_map = dmc_probs.set_index("contig_id")["plas_score"].to_dict()

        out = pred_df.copy()
        out["p_plasmid_dmc"] = out["contig_id"].map(dmc_map).fillna(0.0)
        out["isolated"] = out["contig_id"].map(lambda x: iso.get(x, False))
        rescue_mask = (out["isolated"]) & (out["p_plasmid_dmc"] > plasmid_rescue_threshold) & (out["pred_label"] != "Plasmid")
        out["rescued_plasmid"] = rescue_mask
        out.loc[rescue_mask, "pred_label"] = "Plasmid"
        return out


    def evaluate_predictions(pred_df: pd.DataFrame, gt_file: Path) -> Dict[str, float]:
        gt = pd.read_csv(gt_file).rename(columns={"class_label": "true_label"})[["contig_id", "true_label"]]
        merged = gt.merge(pred_df[["contig_id", "pred_label"]], on="contig_id", how="inner")

        y_true = merged["true_label"]
        y_pred = merged["pred_label"]

        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0
        )

        out = {
            "n_eval": float(len(merged)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(p_weighted),
            "recall_weighted": float(r_weighted),
            "f1_weighted": float(f1_weighted),
        }
        rep = classification_report(y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
        for c in CLASS_ORDER:
            key = c.lower()
            out[f"precision_{key}"] = float(rep.get(c, {}).get("precision", 0.0))
            out[f"recall_{key}"] = float(rep.get(c, {}).get("recall", 0.0))
            out[f"f1_{key}"] = float(rep.get(c, {}).get("f1-score", 0.0))
        return out


    def parse_thresholds(spec: str) -> List[float]:
        spec = spec.strip()
        if "," in spec:
            return [float(x) for x in spec.split(",") if x.strip()]
        if ":" in spec:
            start, stop, step = [float(x) for x in spec.split(":")]
            values = []
            v = start
            while v <= stop + 1e-12:
                values.append(round(v, 6))
                v += step
            return values
        return [float(spec)]


    def run_pipeline(
        dmc_file: Path,
        gfa_file: Path,
        paths_file: Path,
        gt_file: Path,
        output_dir: Path,
        fourcac_script: Path,
        fourcac_env: str,
        asmdir: Path,
        anchor_threshold: float,
        graph_threshold: float,
        plasmid_rescue_threshold: float,
        temperature: float,
    ) -> Dict[str, float]:
        output_dir.mkdir(parents=True, exist_ok=True)

        dmc = load_dmc_probabilities(dmc_file, temperature)
        anchor_stats = write_4cac_input(dmc, output_dir / "scaffolds.fasta.probs_xgb_4class.out", anchor_threshold)

        run_4cac(
            fourcac_script=fourcac_script,
            fourcac_env=fourcac_env,
            asmdir=asmdir,
            classdir=output_dir,
            outdir=output_dir,
            phage_threshold=graph_threshold,
            plasmid_threshold=graph_threshold,
        )

        pred = load_4cac_output(output_dir / "4CAC_classification.fasta")
        contig_nodes_df = read_spades_paths(paths_file)
        adjacency = read_gfa_adjacency(gfa_file)
        pred = apply_plasmid_rescue(pred, dmc, contig_nodes_df, adjacency, plasmid_rescue_threshold)

        pred.to_csv(output_dir / "predictions_hybrid.tsv", sep="\t", index=False)
        dmc.to_csv(output_dir / "dmc_probabilities.tsv", sep="\t", index=False)

        metrics = evaluate_predictions(pred, gt_file)
        metrics["anchor_threshold"] = float(anchor_threshold)
        metrics["graph_threshold"] = float(graph_threshold)
        metrics["temperature"] = float(temperature)
        metrics["plasmid_rescue_threshold"] = float(plasmid_rescue_threshold)
        metrics["anchors_n"] = anchor_stats["anchors_n"]
        metrics["anchors_rate"] = anchor_stats["anchors_rate"]
        metrics["rescued_plasmids"] = float(pred["rescued_plasmid"].sum())

        pd.DataFrame([metrics]).to_csv(output_dir / "metrics_hybrid.tsv", sep="\t", index=False)
        return metrics


    def load_4cac_baseline(c4_file: Path, gt_file: Path) -> Dict[str, float]:
        pred = load_4cac_output(c4_file)
        return evaluate_predictions(pred, gt_file)


    def cmd_compare_baseline(args: argparse.Namespace) -> None:
        base = load_4cac_baseline(Path(args.c4_file), Path(args.gt_file))
        dmc_probs = load_dmc_probabilities(Path(args.dmc_file), args.temperature)
        dmc_pred = dmc_probs[["contig_id", "pred_label_dmc"]].rename(columns={"pred_label_dmc": "pred_label"})
        dmc = evaluate_predictions(dmc_pred, Path(args.gt_file))

        rows = []
        for name, met in [("4cac", base), ("dmc", dmc)]:
            row = {"model": name}
            row.update(met)
            rows.append(row)

        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame(rows)
        out.to_csv(outdir / "baseline_metrics.tsv", sep="\t", index=False)
        print(out[["model", "accuracy", "f1_macro", "f1_plasmid", "recall_plasmid", "precision_virus"]].to_string(index=False))


    def cmd_grid_search(args: argparse.Namespace) -> None:
        out_root = Path(args.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        thresholds = parse_thresholds(args.anchor_thresholds)
        results = []
        for t in thresholds:
            run_dir = out_root / f"thr_{t:.3f}"
            m = run_pipeline(
                dmc_file=Path(args.dmc_file),
                gfa_file=Path(args.gfa_file),
                paths_file=Path(args.paths_file),
                gt_file=Path(args.gt_file),
                output_dir=run_dir,
                fourcac_script=Path(args.fourcac_script),
                fourcac_env=args.fourcac_env,
                asmdir=Path(args.asmdir),
                anchor_threshold=t,
                graph_threshold=args.graph_threshold,
                plasmid_rescue_threshold=args.plasmid_rescue_threshold,
                temperature=args.temperature,
            )
            results.append(m)

        df = pd.DataFrame(results).sort_values(["f1_macro", "f1_plasmid", "accuracy"], ascending=False)
        df.to_csv(out_root / "grid_search_summary.tsv", sep="\t", index=False)
        with (out_root / "best_config.json").open("w") as f:
            json.dump(df.iloc[0].to_dict(), f, indent=2)

        print(df[["anchor_threshold", "f1_macro", "f1_plasmid", "accuracy", "recall_plasmid", "rescued_plasmids"]].head(15).to_string(index=False))


    def cmd_run(args: argparse.Namespace) -> None:
        m = run_pipeline(
            dmc_file=Path(args.dmc_file),
            gfa_file=Path(args.gfa_file),
            paths_file=Path(args.paths_file),
            gt_file=Path(args.gt_file),
            output_dir=Path(args.output_dir),
            fourcac_script=Path(args.fourcac_script),
            fourcac_env=args.fourcac_env,
            asmdir=Path(args.asmdir),
            anchor_threshold=args.anchor_threshold,
            graph_threshold=args.graph_threshold,
            plasmid_rescue_threshold=args.plasmid_rescue_threshold,
            temperature=args.temperature,
        )
        print(pd.DataFrame([m]).to_string(index=False))


    def build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Hybrid DMC + 4CAC propagation")
        sub = p.add_subparsers(dest="command", required=True)

        common = argparse.ArgumentParser(add_help=False)
        common.add_argument("--dmc-file", required=True)
        common.add_argument("--gfa-file", required=True)
        common.add_argument("--paths-file", required=True)
        common.add_argument("--gt-file", required=True)
        common.add_argument("--asmdir", required=True)
        common.add_argument("--fourcac-script", required=True)
        common.add_argument("--fourcac-env", default="4cac_env")
        common.add_argument("--output-dir", required=True)
        common.add_argument("--graph-threshold", type=float, default=0.95)
        common.add_argument("--plasmid-rescue-threshold", type=float, default=0.6)
        common.add_argument("--temperature", type=float, default=1.0)

        p_grid = sub.add_parser("grid-search", parents=[common])
        p_grid.add_argument("--anchor-thresholds", default="0.55:0.95:0.05")
        p_grid.set_defaults(func=cmd_grid_search)

        p_run = sub.add_parser("run", parents=[common])
        p_run.add_argument("--anchor-threshold", type=float, required=True)
        p_run.set_defaults(func=cmd_run)

        p_base = sub.add_parser("compare-baseline")
        p_base.add_argument("--c4-file", required=True)
        p_base.add_argument("--dmc-file", required=True)
        p_base.add_argument("--gt-file", required=True)
        p_base.add_argument("--output-dir", required=True)
        p_base.add_argument("--temperature", type=float, default=1.0)
        p_base.set_defaults(func=cmd_compare_baseline)

        return p


    def main() -> None:
        parser = build_parser()
        args = parser.parse_args()
        args.func(args)


    if __name__ == "__main__":
        main()

