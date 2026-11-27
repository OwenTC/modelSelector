#!/bin/bash
#SBATCH --job-name=synthetic-search
#SBATCH --output=slurm-synthetic-%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec "${PYTHON:-python}" -m examples.synthetic_search.benchmark run "$1" \
    --case "${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch --array}"
