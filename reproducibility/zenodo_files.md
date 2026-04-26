# Zenodo Assets

Zenodo archival package:
- DOI: [10.5281/zenodo.19558452](https://doi.org/10.5281/zenodo.19558452)
- Record: [zenodo.org/records/19558452](https://zenodo.org/records/19558452)

Final paper repo:
- [github.com/scattering/paper-ai-diffraction](https://github.com/scattering/paper-ai-diffraction)

Primary archive:
- `zenodo_paper_repro.tar.gz`

Reviewer archive:
- `reviewer_compact_assets.tar.gz`

Archive contents:
- final-stage ViT checkpoints named in the manuscript
- supplemental VT checkpoints
- positional-ablation checkpoints
- curated CNN checkpoints for the retained PRX CNN figures and tables
- `CNN_README.md`
- `cnn_artifact_map.csv`
- recovered CNN Fig. S2 checkpoint `xrd_resnet_rbwbgj89.pth`
- compact result JSONs (under `results/`)
- paper configs
- canonical and archived launchers
- short reproducibility notes

Supplemental notebook assets (`assets/figure_data/`) are **bundled in the git repo**, not in the Zenodo archive:
  - `assets/figure_data/1k_structures.csv` — 1,000 crystal structures for Fig S5 notebook
  - `assets/figure_data/interp_metadata_clean.csv` — HDF5-index-to-structure_id map for Fig S5 notebook
  - `assets/figure_data/conf_mat_top5_copper-sweep-1_table_103_ff53214644fd32c50e63.table.json` — W&B confusion matrix table artifact for Fig S3 notebook
- compact reviewer assets, packaged separately in `reviewer_compact_assets.tar.gz`:
  - `assets/reviewer_examples/correct_case_015_Arsenopyrite__R050071-1__6130.csv`
  - `assets/reviewer_examples/correct_case_015_Arsenopyrite__R050071-1__6130.json`
  - `assets/reviewer_examples/failure_case_000_Actinolite__R050336-1__5330.csv`
  - `assets/reviewer_examples/failure_case_000_Actinolite__R050336-1__5330.json`
  - `assets/reviewer_examples/reviewer_case_metadata.csv`
  - `assets/reviewer_examples/manifest.json`
  - `results/reviewer/ext_group_priors.csv`
  - `results/reviewer/ext_group_priors.json`
  - `results/reviewer/rruff325_precomputed_inference.json`

These reviewer assets are small, derived, and paper-backed. They are intended to support the public notebook walkthrough without redistributing the full RRUFF benchmark.

Archive split note:
- code, notebooks, and paper-facing scripts stay in the GitHub repo
- Zenodo carries checkpoints, compact derived artifacts, configs, launchers, and
  short notes
- reviewer assets remain packaged separately in `reviewer_compact_assets.tar.gz`

Benchmark note:
- this package releases the paper-facing benchmark-construction scripts and documentation
- it does not redistribute the upstream RRUFF-derived benchmark HDF5s or raw source files
- see [docs/BENCHMARKS.md](../docs/BENCHMARKS.md)

Expected local placement for downloaded checkpoints:

```text
external/checkpoints/
```

Expected workflow:
1. Download the Zenodo archive or selected checkpoint files.
2. Place the checkpoint files under `external/checkpoints/`.
3. Use [checkpoint_manifest.csv](checkpoint_manifest.csv) to match every manuscript-named checkpoint to its archival filename.
4. Use [dataset_manifest.csv](dataset_manifest.csv) to provide required external benchmark and trainready datasets.

Reviewer workflow from the archival package:
1. Extract `reviewer_compact_assets.tar.gz` into the repo root, or inspect the same files directly in the GitHub repo.
2. Use the shipped reviewer example CSVs and metadata for the notebook walkthrough.
3. Use the compact reviewer priors and precomputed `RRUFF-325` summary JSON for notebook browsing.
4. Use checkpoints from Zenodo for live single-pattern inference.
5. Do not expect the full RRUFF benchmark HDF5s to be redistributed.

Published Zenodo identifiers:
1. version DOI: [10.5281/zenodo.19558452](https://doi.org/10.5281/zenodo.19558452)
2. concept DOI: [10.5281/zenodo.19558451](https://doi.org/10.5281/zenodo.19558451)
3. record URL: [zenodo.org/records/19558452](https://zenodo.org/records/19558452)
