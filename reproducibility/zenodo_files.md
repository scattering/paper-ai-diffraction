# Zenodo Assets

Current Zenodo archival package:
- DOI: [10.5281/zenodo.21043734](https://doi.org/10.5281/zenodo.21043734)
- Record: [zenodo.org/records/21043734](https://zenodo.org/records/21043734)
- Concept DOI: [10.5281/zenodo.19558451](https://doi.org/10.5281/zenodo.19558451)

Final paper repo:
- [github.com/scattering/paper-ai-diffraction](https://github.com/scattering/paper-ai-diffraction)

Primary archive:
- `paper_ai_diffraction_revised_20260629.tar.gz`

Current archive contents:
- current final-stage flat-37 ViT checkpoints named in the manuscript
- matched regular-transformer architecture-control checkpoint
- compact result JSON/CSV artifacts under `results/`
- RRUFF-325 and RRUFF-473 case-ID CSVs under `benchmark_ids/`
- paper configs
- canonical and archived launchers
- short reproducibility notes

For the current paper package, the active final-stage ViT release surface is:

- `xrd_model_pubfix_flat37_20260622_stage2c_r2346k_s1337_best.pth`
- `xrd_model_pubfix_flat37_20260622_stage1_u2m_s1337_best.pth`
- `xrd_model_pubfix_flat37_20260622_po200k_s1337_best.pth`
- `xrd_model_pubfix_flat37_20260622_dualsource2500k_s1337_best.pth`
- `configs/flat37/`
- `assets/lookups/eg37_mapping_20260622.{json,csv}`
- `results/flat37/`
- the matched SG-to-EG compact outputs
- `xrd_model_pubfix_rt_flat37_20260624_stage2c_r2346k_s1337_best.pth`
- the matched regular-transformer compact outputs
- `benchmark_ids/rruff325_case_ids.csv`
- `benchmark_ids/rruff473_case_ids.csv`
- compact information-theory JSON summaries; posterior `.npz` files are not part
  of the main Zenodo package

WiSE-FT checkpoints are historical or exploratory archive entries, not part of the current-paper final model package.

Supplemental notebook assets (`assets/figure_data/`) are **bundled in the git repo**, not in the Zenodo archive:
  - `assets/figure_data/1k_structures.csv` — 1,000 crystal structures for Fig S5 notebook
  - `assets/figure_data/interp_metadata_clean.csv` — HDF5-index-to-structure_id map for Fig S5 notebook
  - `assets/figure_data/conf_mat_top5_copper-sweep-1_table_103_ff53214644fd32c50e63.table.json` — W&B confusion matrix table artifact for Fig S3 notebook
- compact reviewer assets are bundled in the GitHub repo:
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
- reviewer assets remain in the GitHub repo
- current release status is tracked in [CURRENT_PAPER_RELEASE_NOTES_2026-06-29.md](CURRENT_PAPER_RELEASE_NOTES_2026-06-29.md)

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

Reviewer workflow:
1. Inspect the reviewer example files directly in the GitHub repo.
2. Use the shipped reviewer example CSVs and metadata for the notebook walkthrough.
3. Use the compact reviewer priors and precomputed `RRUFF-325` summary JSON for notebook browsing.
4. Use checkpoints from Zenodo for live single-pattern inference.
5. Do not expect the full RRUFF benchmark HDF5s to be redistributed.

Published Zenodo identifiers:
1. version DOI: [10.5281/zenodo.21043734](https://doi.org/10.5281/zenodo.21043734)
2. concept DOI: [10.5281/zenodo.19558451](https://doi.org/10.5281/zenodo.19558451)
3. record URL: [zenodo.org/records/21043734](https://zenodo.org/records/21043734)
4. previous version: [10.5281/zenodo.19558452](https://doi.org/10.5281/zenodo.19558452)
