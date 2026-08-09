# Journal open-source release checklist

## Code hygiene

- [x] Replace `/mimer/...`, `/nobackup/...`, `/home/...`, and Slurm-account defaults with relative paths or CLI arguments.
- [x] Remove notebook checkpoints, `__pycache__`, backup files, cluster logs, and local virtual environments from the release archive.
- [x] Add a one-command smoke test for source compilation and release-file checks (`scripts/smoke_test.sh`).
- [ ] Record Python, PyTorch, CUDA, NetCDF, Cartopy, and `cnmaps` versions.
- [x] Run `python -m compileall src_2` and document any optional dependencies.

## Data and models

- [ ] Decide whether raw ERA5 files are redistributed or downloaded from the official source.
- [ ] Publish checksums and immutable revisions for every data/checkpoint archive.
- [ ] Include coordinate order, units, normalization statistics, time split, and sample indexing rules.
- [ ] Verify that all model checkpoints and administrative boundary data are redistributable.

## Paper reproducibility

- [ ] Map every manuscript figure/table to one public command and one output path.
- [ ] Record seeds, city list, model list, input/prediction lengths, patch size, and evaluation metrics.
- [ ] Complete `CITATION.cff.template`, add the final article DOI, and include the repository DOI.
- [ ] Add a release tag matching the version cited by the article.
