#!/bin/bash
#SBATCH -A YOUR_PROJECT_ACCOUNT
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --array=1-9
#SBATCH --job-name=som_tc_sv
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs_temporal_spatial/cluster_%A_%a.out
#SBATCH --error=logs_temporal_spatial/cluster_%A_%a.err

set -euo pipefail

PROJECT_DIR=${WIND_PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}
PYTHON_BIN=${PROJECT_DIR}/venv_gh200/bin/python

cd "${PROJECT_DIR}"

echo "Node: $(hostname)"
echo "Cluster: ${SLURM_ARRAY_TASK_ID}"
echo "Start: $(date)"

"${PYTHON_BIN}" src_2/drawing_temporal_spatial_consistency.py \
    --cluster "${SLURM_ARRAY_TASK_ID}" \
    --samples 30

echo "End: $(date)"
