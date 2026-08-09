#!/bin/bash
#SBATCH -A YOUR_PROJECT_ACCOUNT
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --job-name=setup_plot_env
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs_temporal_spatial/setup_%j.out
#SBATCH --error=logs_temporal_spatial/setup_%j.err

set -euo pipefail

PROJECT_DIR=${WIND_PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}
VENV_DIR=${PROJECT_DIR}/venv_gh200

echo "Node architecture: $(uname -m)"
echo "Python: $(command -v python3)"
python3 --version

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install numpy pandas xarray matplotlib netCDF4 'dask[array]'
"${VENV_DIR}/bin/python" -c "import numpy, matplotlib, xarray; print('GH200 environment OK')"
