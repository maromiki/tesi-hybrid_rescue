# Hybrid + Rescue: Multi-class Metagenomic Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Data Analysis](https://img.shields.io/badge/Data%20Analysis-Pandas%20%7C%20Numpy-green)
![HPC](https://img.shields.io/badge/HPC-SLURM-orange)
![Biology](https://img.shields.io/badge/Domain-Metagenomics-red)

## 📌 Executive Summary
This repository contains the source code for the **Hybrid + Rescue** computational model, developed to optimize multi-class metagenomic classification. 

The pipeline addresses a major bottleneck in R&D bioinformatics: the trade-off between semantic sensitivity and topological precision in assembly graphs. By integrating Convolutional Neural Networks (CNNs) via DeepMicroClass (DMC) with topological propagation via 4CAC, this architecture maximizes classification accuracy. 

A novel biological heuristic, **Plasmid Rescue**, was designed and implemented to systematically recover plasmid contigs that are typically lost as isolated nodes in standard topological graph filtering.

## 🧬 Biological Rationale & Architecture
The system integrates two orthogonal paradigms:
1. **Semantic Extraction:** Utilizing Softmax probability vectors from DMC.
2. **Topological Propagation:** Mapping predictions onto the assembly graph via 4CAC.
3. **Plasmid Rescue Heuristic:** A custom conditional algorithm to recover biologically relevant but topologically isolated circular DNA fragments.

## 📂 Repository Structure
*   `scripts/`: Core source code and Python scripts.
    *   `hybrid_dmc_4cac.py`: Main orchestrator (CNN-Graph integration, metrics calculation, Plasmid Rescue logic).
    *   `evaluate_sharon_strategies.py`: Benchmarking script for environmental datasets.
*   `config/`: JSON configuration files for reproducible input paths.
*   `docs/`: Analysis and validation reports.
*   `results/`: Output directories for pipeline metrics.
*   `validazione/`: HPC/SLURM scripts for scaling the pipeline on computing clusters.

## ⚙️ Requirements & Installation
It is highly recommended to run this pipeline within a virtual environment (e.g., Conda).

**Core Dependencies:**
*   `numpy`
*   `pandas`
*   `scikit-learn`
*   Configured environment for 4CAC (specified via `--fourcac-env` flag).

## 🚀 Usage & Pipeline Execution

The primary orchestrator `hybrid_dmc_4cac.py` handles the data flow. 

**1. Baseline Comparison (4CAC vs DMC)**
Evaluates the isolated performance of the base models prior to hybrid integration.
```
python scripts/hybrid_dmc_4cac.py compare-baseline \
  --c4-file data/output/4cac/sharon/4CAC_classification.fasta \
  --dmc-file data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --gt-file data/output/sharon/sharon_ground_truth.csv \
  --output-dir results/baseline
```

**2. Parameter Optimization (Grid Search)**
Executes a parameter sweep to assess the impact of thresholds on classification metrics.
```
python scripts/hybrid_dmc_4cac.py grid-search \
  --dmc-file data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --gfa-file data/output/metaspades/sharon/assembly_graph_with_scaffolds.gfa \
  --paths-file data/output/metaspades/sharon/scaffolds.paths \
  --gt-file data/output/sharon/sharon_ground_truth.csv \
  --output-dir results/grid_search \
  --fourcac-script path/to/4CAC/run_4CAC.py \
  --asmdir data/output/metaspades/sharon/ \
  --anchor-thresholds 0.55:0.95:0.05 \
  --plasmid-rescue-threshold 0.6 \
  --temperature 1.0 
```

## 📊 Expected Outputs
Upon successful execution, the pipeline generates:
*   `predictions_hybrid.tsv`: Final taxonomic labels assigned to each contig.
*   `dmc_probabilities.tsv`: Normalized Softmax vectors extracted from DMC.
*   `metrics_hybrid.tsv`: Performance metrics (Precision, Recall, F1-score).
*   `grid_search_summary.tsv`: Comparative table of tested configurations.

## 🖥️ HPC Cluster Deployment (SLURM)
The repository includes the infrastructure to scale the pipeline on High-Performance Computing (HPC) clusters. It handles both short-read (metaSPAdes) and long-read (Flye) assemblies.

To submit the workflow to a SLURM cluster:
```
bash slurm/contig_scenarios/submit_workflow.sh
```

---
*Project developed as the MSc Thesis for Industrial Biotechnology at the University of Padova.*
