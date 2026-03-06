#!/bin/bash
#SBATCH --job-name=assemble_long_flye
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=36:00:00
#SBATCH --output=.slurm/contig_scenarios/logs/%x_%j.out
#SBATCH --error=.slurm/contig_scenarios/logs/%x_%j.err

set -euo pipefail

source data/miniconda3/etc/profile.d/conda.sh
conda activate flye

READS=".validazione/camisim_contig_scenarios/runs/long/camisim/sample_0/reads/anonymous_reads.fq.gz"
OUT_DIR=".validazione/camisim_contig_scenarios/runs/long/assembly_flye"
SCRATCH="/ext/${USER}/flye_${SLURM_JOB_ID}"

mkdir -p "${OUT_DIR}" "${SCRATCH}"
cp "${READS}" "${SCRATCH}/reads.fq.gz"

flye \
  --nano-raw "${SCRATCH}/reads.fq.gz" \
  --meta \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --out-dir "${SCRATCH}/flye_out"

cp "${SCRATCH}/flye_out/assembly.fasta" "${OUT_DIR}/"
cp "${SCRATCH}/flye_out/assembly_graph.gfa" "${OUT_DIR}/"
cp "${SCRATCH}/flye_out/assembly_info.txt" "${OUT_DIR}/"
cp "${SCRATCH}/flye_out/flye.log" "${OUT_DIR}/"

rm -rf "${SCRATCH}"
echo "DONE long assembly -> ${OUT_DIR}"
