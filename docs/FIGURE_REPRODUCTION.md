# Figure-to-script map

This map is intended to become the figure reproduction table in the journal supplement. Commands should be finalized only after the public data/checkpoint revisions are frozen.

| Analysis product | Primary entry point | Required inputs | Current local output |
|---|---|---|---|
| SOM geographic clusters | `src_2/drawing_som.py` | `som_map.npy`, `lat_lon_grid.npz`, `cnmaps`/Cartopy data | `som_vis/` |
| SOM cluster temporal/spatial consistency | `src_2/drawing_temporal_spatial_consistency.py` | processed NetCDF, SOM map, grid metadata | `som_post_analysis/individual_clusters/` |
| Model time-series comparison | `src_2/comparison.py` and `src_2/evaluation/vis.py` | per-model prediction/true arrays | `24-24/*/analysis_results_*` and `7-24/*/analysis_results_*` |
| Multi-city comparison panels | `src_2/drawing_exp.py` | per-city model arrays | `24-24/combined_regions_comparison.png`, `7-24/combined_regions_comparison.png` |
| Prediction field panels and animations | `src_2/analysis_pred.py`, `src_2/prediction_region_gif.py` | checkpoints or cached predictions, grid metadata | `prediction_region_animations_24-24/` and local visualization folders |
| PSD analysis | `analysis_PSD.py` | prediction/observation time series | `psd_7_24/` |
| Statistical/model summaries | `analysis_result.py`, `analysis_rmse.py`, `analysis_tri.py`, `anova.py` | metrics CSV files | `aggregated_results/`, `fig/` |
| A14/grid skill maps | `src_2/a14_grid_skill_maps.py`, `src_2/a14_vis.py` | model results, SOM/grid metadata | `fig/a14_grid_skill/` |

Before publication, each row should have a tested command, a data revision, a checkpoint revision, and an expected-output checksum.
