#!/bin/bash
#SBATCH --job-name=camisim_short_meta
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=.slurm/contig_scenarios/logs/%x_%j.out
#SBATCH --error=.slurm/contig_scenarios/logs/%x_%j.err

set -euo pipefail

source data/miniconda3/etc/profile.d/conda.sh
conda activate camisim

# CAMISIM/Nextflow crea env Conda dinamici; con channel_priority strict
# biopython può risultare irrisolvibile (bioconda vs conda-forge).
conda config --set channel_priority flexible
export CONDA_CHANNEL_PRIORITY=flexible

CAMISIM_DIR="tools/CAMISIM"
DATASET_DIR="data/dataset/camisim"
OUT_DIR=".validazione/camisim_contig_scenarios/runs/short/camisim"
DIST_FILE="${DATASET_DIR}/distribution_0.txt"
SCRATCH_ROOT="/ext/${USER}/finalissima_camisim_short_${SLURM_JOB_ID}"
WORK_DIR="${SCRATCH_ROOT}/nxf_work"
TMP_OUT_DIR="${SCRATCH_ROOT}/camisim_out"

mkdir -p "${OUT_DIR}"
mkdir -p "${WORK_DIR}"
mkdir -p "${TMP_OUT_DIR}"
mkdir -p ".slurm/contig_scenarios/logs"

nextflow run "${CAMISIM_DIR}/main.nf" \
  -w "${WORK_DIR}" \
  --pipeline metagenomic \
  --outdir "${TMP_OUT_DIR}" \
  --number_of_samples 1 \
  --size '1.0' \
  --type 'art' \
  --gsa true \
  --profile_read_length 150 \
  --fragment_size_mean 350 \
  --fragment_size_sd 50 \
  --genomes_total 600 \
  --genomes_real 600 \
  --strain_simulation_template "${CAMISIM_DIR}/defaults/template.tree" \
  --metadata_file "${DATASET_DIR}/metadata.tsv" \
  --genome_locations_file "${DATASET_DIR}/genome_locations.tsv" \
  --ncbi_taxdump "${DATASET_DIR}/taxonomy/" \
  --distribution_files "${DIST_FILE}"

mkdir -p "${OUT_DIR}/sample_0/reads" "${OUT_DIR}/pooled_gsa"
cp "${TMP_OUT_DIR}/sample_0/reads/anonymous_reads.fq.gz" "${OUT_DIR}/sample_0/reads/"
cp "${TMP_OUT_DIR}/pooled_gsa/gsa_pooled.fasta.gz" "${OUT_DIR}/pooled_gsa/"
cp "${TMP_OUT_DIR}/gsa_pooled_mapping.tsv.gz" "${OUT_DIR}/"
cp "${TMP_OUT_DIR}/taxonomic_profile_0.txt" "${OUT_DIR}/"

rm -rf "${SCRATCH_ROOT}"

echo "DONE short CAMISIM metagenomic run -> ${OUT_DIR}"
