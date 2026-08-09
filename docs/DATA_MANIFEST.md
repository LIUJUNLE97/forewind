# Data and artifact manifest

This inventory describes the current working tree as of 2026-08-09. Sizes are approximate and are intended for release planning, not as checksums.

| Local path | Approx. size | Role | Recommended publication treatment |
|---|---:|---|---|
| `processed/` | 61 GB | ERA5-derived monthly NetCDF inputs | Publish as a separate versioned data archive or provide the official upstream download recipe. |
| `npy_data/` | 154 GB | normalized arrays, timestamps, and statistics used by training | Do not put in Git; publish a compressed, versioned archive only if redistribution is permitted. |
| `24-24/` | 2.8 GB | checkpoints, metrics, and experiment outputs | Publish selected checkpoints/metrics separately; keep raw run logs out of the code repository. |
| `7-24/` | 4.7 GB | checkpoints, metrics, and experiment outputs | Same treatment as `24-24/`; document missing/failed model-city combinations. |
| `models/` | 6.8 MB | auxiliary model files | Check provenance and include only files needed for reproduction. |
| `som_map.npy` | 288 KB | 3×3 SOM labels | Suitable for the lightweight metadata archive. |
| `lat_lon_grid.npz` | 4 KB | latitude/longitude grid | Suitable for the lightweight metadata archive. |
| `fig/`, `som_vis/`, `som_post_analysis/` | tens of MB | generated figures and reports | Publish selected final figures as a separate artifact; do not mix them with source code. |
| `.runtime_*`, `venv*` | 7+ GB | machine-specific environments | Never publish; replace with dependency files and environment documentation. |

## Minimum reproducibility bundle

The smallest useful public data record should contain:

1. `lat_lon_grid.npz`, the final `som_map.npy`, and normalization statistics;
2. a manifest of the processed ERA5 variables, coordinate order, units, time range, and missing-value handling;
3. one immutable checkpoint per reported model/configuration, or a documented training command to recreate it;
4. metrics tables and the exact Git commit used to generate the manuscript figures.

## Data licensing checks

ERA5/Copernicus terms, Hugging Face dataset permissions, administrative boundary data (`cnmaps`/Natural Earth), fonts, and all pretrained/checkpoint files must be checked individually. The source-code MIT license does not automatically license these artifacts.
