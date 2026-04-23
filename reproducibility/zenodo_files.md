# Zenodo Assets

Zenodo archival package:
- pending public record / DOI

Final paper repo:
- [github.com/scattering/paper-ai-diffraction](https://github.com/scattering/paper-ai-diffraction)

Primary archive:
- `zenodo_paper_repro.tar.gz`

Reviewer archive:
- `reviewer_compact_assets.tar.gz`

Archive contents:
- all checkpoints explicitly named in the manuscript main text or supplement
- compact result JSONs
- paper configs
- canonical and archived launchers
- short reproducibility notes
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

Benchmark note:
- this package releases the paper-facing benchmark-construction scripts and documentation
- it does not redistribute the upstream RRUFF-derived benchmark HDF5s or raw source files
- see [docs/BENCHMARKS.md](/tmp/paper-ai-diffraction-fresh/docs/BENCHMARKS.md)

Expected local placement for downloaded checkpoints:

```text
external/checkpoints/
```

Expected workflow:
1. Download the Zenodo archive or selected checkpoint files.
2. Place the checkpoint files under `external/checkpoints/`.
3. Use [checkpoint_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/checkpoint_manifest.csv) to match every manuscript-named checkpoint to its archival filename.
4. Use [dataset_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/dataset_manifest.csv) to provide required external benchmark and trainready datasets.

Reviewer workflow from the archival package:
1. Extract `reviewer_compact_assets.tar.gz` into the repo root, or inspect the same files directly in the GitHub repo.
2. Use the shipped reviewer example CSVs and metadata for the notebook walkthrough.
3. Use the compact reviewer priors and precomputed `RRUFF-325` summary JSON for notebook browsing.
4. Use checkpoints from Zenodo for live single-pattern inference.
5. Do not expect the full RRUFF benchmark HDF5s to be redistributed.

After Zenodo publication, update this file with:
1. the final DOI
2. the final record URL
3. any article DOI link
