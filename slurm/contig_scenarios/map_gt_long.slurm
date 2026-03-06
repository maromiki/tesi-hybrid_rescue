#!/bin/bash
#SBATCH --job-name=map_gt_long
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=.slurm/contig_scenarios/logs/%x_%j.out
#SBATCH --error=.slurm/contig_scenarios/logs/%x_%j.err

set -euo pipefail

source data/miniconda3/etc/profile.d/conda.sh
conda activate minimap2
PYTHON_BIN="data/miniconda3/envs/thesis_env/bin/python"

RUN_DIR=".validazione/camisim_contig_scenarios/runs/long"
ASM="${RUN_DIR}/assembly_flye/assembly.fasta"
GSA_REF="${RUN_DIR}/camisim/pooled_gsa/gsa_pooled.fasta.gz"
GSA_MAP="${RUN_DIR}/camisim/gsa_pooled_mapping.tsv.gz"
PAF="${RUN_DIR}/assembly_flye/assembly_vs_gsa.paf"
GT_OUT="${RUN_DIR}/ground_truth/assembly_gt.tsv"
TRUE_GSA_GT="${RUN_DIR}/ground_truth/contig_truth.tsv"

mkdir -p "${RUN_DIR}/ground_truth"

"${PYTHON_BIN}" data/script/camisim/build_contig_gt.py \
  --mapping "${GSA_MAP}" \
  --reference-map data/dataset/camisim/reference_class_map.tsv \
  --out-dir "${RUN_DIR}/ground_truth"

minimap2 -x asm20 -t "${SLURM_CPUS_PER_TASK}" "${GSA_REF}" "${ASM}" > "${PAF}"

"${PYTHON_BIN}" data/script/camisim/map_contigs_to_gsa_gt.py \
  --paf "${PAF}" \
  --assembly "${ASM}" \
  --true-gsa-gt "${TRUE_GSA_GT}" \
  --gsa-mapping "${GSA_MAP}" \
  --output "${GT_OUT}" \
  --min-identity 0.85 \
  --min-cov 0.50 \
  --min-length 500

echo "DONE long GT mapping -> ${GT_OUT}"
