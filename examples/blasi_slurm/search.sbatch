#!/bin/bash
#SBATCH --job-name=blasi-exact-search
#SBATCH --output=slurm-blasi-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@120

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

exec "${PYTHON:-python}" examples/blasi_slurm/run.py "$@"
