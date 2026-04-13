# Hybrid + Rescue: Multi-Class Metagenomic Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Data Analysis](https://img.shields.io/badge/Data%20Analysis-Pandas%20%7C%20Scikit--Learn-green)
![HPC](https://img.shields.io/badge/HPC-SLURM-orange)
![Biology](https://img.shields.io/badge/Domain-Metagenomics-red)
![Genomics](https://img.shields.io/badge/Genomics-SPAdes%20%7C%20Flye-red)

## 📌 Executive Summary
**Hybrid + Rescue** is a reproducible, high-performance computational pipeline designed to optimize multi-class metagenomic classification. 

The architecture bridges the gap between semantic sensitivity and topological precision by combining **DeepMicroClass (DMC)** probabilistic predictions with **4CAC-style** topological graph propagation. To overcome the inherent limitations of rigid graph filtering, the pipeline integrates a novel biological heuristic—**Plasmid Rescue**—to systematically recover isolated plasmid contigs.

```markdown
> **Methodology & Authorship Note:** The biological logic (e.g., the *Plasmid Rescue* heuristic), the pipeline architecture, and the HPC/SLURM environment orchestration are my original work. The Python/R scripting and code implementation were developed using AI-assisted programming (LLMs) under my direct logical supervision and validation.
```

### 🧠 Core Logic & Current Behavior
*   **Non-Destructive Thresholding:** Contigs falling below the anchor threshold ("uncertain") are not hard-coded/overwritten with uniform probabilities (0.25). Instead, their original DMC probability vectors are preserved and propagated.
*   **Diagnostic Tuning:** Threshold adjustments dynamically update diagnostic metrics (anchor counts/rates) without forcing the alteration of uncertain-contig class scores, ensuring robust and transparent data flows.

---

## 📂 Repository Structure
*   `scripts/`: Core Python modules and pipeline orchestrators.
    *   `hybrid_dmc_4cac.py`: Main execution script (baseline, grid-search, tuning, final run).
    *   `evaluate_sharon_strategies.py`: Benchmarking on environmental datasets.
*   `config/`: JSON configuration files for path management (e.g., `sharon_paths.json`).
*   `slurm/`: Job submission scripts for HPC cluster deployment.
*   `validation/`: Output directories for CAMISIM simulated scenarios.
*   `docs/`: Technical integration summaries (`TOOL_ANALYSIS.md`).
*   `results/`: Validation and metrics outputs.

## ⚙️ Requirements
Ensure a Python virtual environment (e.g., Conda) is active.
*   `numpy`
*   `pandas`
*   `scikit-learn`

---

## 🚀 Execution Modules

The `hybrid_dmc_4cac.py` orchestrator handles multiple execution modes.

### 1. Baseline Comparison
Evaluates isolated base models (4CAC vs DMC) prior to hybrid integration.
```
python scripts/hybrid_dmc_4cac.py compare-baseline \
  --c4-file data/output/4cac/sharon/4CAC_classification.fasta \
  --dmc-file data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --gt-file data/output/sharon/sharon_ground_truth.csv \
  --output-dir results/baseline
```

### 2. Anchor Threshold Search (Grid Search)
Performs parameter sweeps for optimization. *Note: With current logic, this is primarily diagnostic to track anchor rates.*
```
python scripts/hybrid_dmc_4cac.py grid-search \
  --dmc-file data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --gfa-file data/output/metaspades/sharon/assembly_graph_with_scaffolds.gfa \
  --paths-file data/output/metaspades/sharon/scaffolds.paths \
  --gt-file data/output/sharon/sharon_ground_truth.csv \
  --output-dir results/grid_search \
  --anchor-thresholds 0.55:0.95:0.05 \
  --plasmid-rescue-threshold 0.6 \
  --temperature 1.0 \
  --alpha 0.65 \
  --n-iter 20
```

### 3. Final Run (Optimized Thresholds)
Deploys the pipeline using the best parameters identified during tuning.
```
python scripts/hybrid_dmc_4cac.py run \
  --dmc-file data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --gfa-file data/output/metaspades/sharon/assembly_graph_with_scaffolds.gfa \
  --paths-file data/output/metaspades/sharon/scaffolds.paths \
  --gt-file data/output/sharon/sharon_ground_truth.csv \
  --output-dir results/final_model \
  --anchor-threshold <BEST> \
  --plasmid-rescue-threshold 0.6 \
  --temperature 1.0 \
  --alpha 0.65 \
  --n-iter 20
```

### 4. Historical Strategy Comparison (4-Class Tuning)
Comprehensive benchmarking across baseline and hybrid strategies.
```
python scripts/evaluate_sharon_strategies.py \
  --gt data/output/sharon/sharon_ground_truth.csv \
  --dmc data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --hyb data/output/hybrid/sharon/4CAC_classification.fasta \
  --c4 data/output/4cac/sharon/4CAC_classification.fasta \
  --circular data/output/metaspades/sharon/circular_contigs_filtered.txt \
  --out results/strategy_comparison.tsv
```

---

## 📊 Expected Outputs
*   `predictions_hybrid.tsv`: Final taxonomic label per contig.
*   `node_state.tsv`: Detailed node states and propagated probabilities.
*   `metrics_hybrid.tsv`: Full metrics + diagnostics (`anchors_n`, `anchors_rate`, `rescued_plasmids`).
*   `predictions_best_tuned.tsv`: Predictions from the accuracy-tuned model.

---

## 🖥️ HPC & CAMISIM Validation Workflow (SLURM)
To ensure robustness across sequencing technologies, the repository includes a controlled contig-level benchmarking workflow scaled via **SLURM**.

The pipeline orchestrates two parallel environments:
1.  **Short-read branch:** CAMISIM (`art`) simulation + **metaSPAdes** (`--meta`) assembly.
2.  **Long-read branch:** CAMISIM (`nanosim3`) simulation + **Flye** (`--meta`) assembly.

**Submit the automated workflow to the cluster:**
```
bash slurm/contig_scenarios/submit_workflow.sh
```
*Generated validation outputs and plots are automatically routed to `validation/camisim_contig_scenarios/` under their respective `short/` and `long/` subdirectories.*

---

*Project developed as the MSc Thesis for Industrial Biotechnology at the University of Padova.*

