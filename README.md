# Forewind

Code and analysis workflows for SOM-based regional wind-field prediction with CNN, FNO, KAN, ConvLSTM, ConvGRU, TrajGRU, PredRNN, and U-Net baselines.

This repository is being prepared as the reproducibility package for the accompanying journal article. It contains the source code and lightweight metadata needed to reproduce the analysis. The large ERA5 files, model checkpoints, virtual environments, and generated figures are intentionally kept outside version control.

## Repository layout

```text
src_2/                         training, evaluation, SOM, PSD, and plotting code
configs/                       example release configuration and path conventions
data/metadata/                 small SOM/grid/normalization metadata for release
docs/                          data inventory and release checklist
24-24/, 7-24/                  local experiment outputs (not part of the source release)
processed/, npy_data/          local ERA5-derived data (not part of the source release)
prediction_region_animations/  local GIF/AVI products (not part of the source release)
```

The two experiment directories correspond to the 24-hour and 7-hour input configurations used during development. Their names are retained for traceability; the exact train/validation/test settings should be recorded in the article supplement before publication.

## Installation

Use a clean Python environment. Plotting and post-processing require:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-plot.txt
```

Training additionally requires a platform-appropriate PyTorch installation and the packages listed in `requirements-training.txt`.

The code was developed on Linux with Python 3.9/3.11. GPU training requires a CUDA-compatible PyTorch build; CPU-only plotting does not.

## Data

The workflow expects ERA5-derived NetCDF files, a normalized NumPy array, timestamps, normalization statistics, and a 3×3 SOM map. The exact inventory and recommended publication tiers are documented in [docs/DATA_MANIFEST.md](docs/DATA_MANIFEST.md).

The helper `download_data.py` downloads the processed ERA5 dataset from the configured Hugging Face dataset. Before public release, verify that the ERA5/Copernicus attribution and redistribution terms are included in the data record and article supplement.

## Main workflows

All commands should be run from the project root with `src_2` on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PWD/src_2:${PYTHONPATH:-}"
python src_2/data_process_kit/nc2npy.py --help
python src_2/run_models.py --help
python src_2/comparison.py --help
python src_2/drawing_temporal_spatial_consistency.py --help
python src_2/prediction_region_gif.py --help
```

The scripts accept explicit data/output paths where possible. Do not rely on the historical cluster paths embedded in legacy batch files; replace them with paths under the local checkout or pass the corresponding command-line arguments.

## Demos
<img width="1440" height="900" alt="shanghai_sample_0_model_comparison" src="https://github.com/user-attachments/assets/31f5864a-29df-4808-8b41-3197a25c0a9a" />
<img width="1440" height="900" alt="wuhan_sample_0_model_comparison" src="https://github.com/user-attachments/assets/4520c074-1197-4294-8492-8f9b9bdf1f76" />
<img width="1440" height="900" alt="golmud_sample_0_model_comparison" src="https://github.com/user-attachments/assets/0d2b6839-585c-49e9-bcf0-09a3b5d35f36" />
<img width="1440" height="900" alt="chengdu_sample_0_model_comparison" src="https://github.com/user-attachments/assets/d6b7c1e5-8d58-4586-a159-ec4caacf19e9" />
x

## Reproducibility policy

- Publish immutable data/checkpoint archives separately (Zenodo, Hugging Face, or the journal repository) and record their DOI/revision in the final README.
- Record the exact Git commit, Python version, PyTorch/CUDA version, random seed, data revision, and command line for every reported result.
- Confirm that all third-party datasets, boundary files, fonts, and pretrained checkpoints can legally be redistributed.

## License and citation

The source code is released under the MIT License; see [LICENSE](LICENSE). Citation metadata is provided as a template in [CITATION.cff.template](CITATION.cff.template) and must be completed with the final author list, article title, DOI, and repository URL before publication.
