# Machine-specific path audit

The current working tree still contains historical absolute paths. This is a release blocker because a clean checkout will otherwise try to read cluster-local directories.

## Highest-priority files

- `src_2/run_models.py`, `src_2/eval.py`, `src_2/eval_slice.py`, and `src_2/analysis_pred.py`: training/evaluation defaults.
- `src_2/data_process_kit/*.py`: preprocessing, SOM, and statistics defaults.
- `src_2/a14_grid_skill_maps.py` and `src_2/a14_vis.py`: supplementary skill-map workflow.
- `analysis_result.py`, `analysis_tri.py`, `anova.py`, `prob.py`, and `persistence.py`: root-level analysis scripts.
- `module.sh`, `submit_a14_grid_skill.sh`, `submit_temporal_spatial_plots.sh`, and `logs_a14_*/`: cluster submission examples.

## Required release treatment

1. Replace absolute defaults with paths relative to `WIND_PROJECT_ROOT` or explicit command-line arguments.
2. Keep Slurm scripts in an optional `hpc/` directory and replace account, partition, and environment values with placeholders.
3. Remove personal font paths; `WIND_PROJECT_ARIAL_FONT` is now supported by the main plotting modules.
4. Re-run the path scan below before tagging a release:

```bash
rg -n '(/mimer/NOBACKUP|/nobackup/proj|/home/)' \
  --glob '*.py' --glob '*.sh' \
  --glob '!venv*/**' --glob '!.runtime*/**' .
```

The scan should return zero machine-specific paths in the public code path. Historical cluster scripts may be retained separately as clearly marked, non-portable examples.
