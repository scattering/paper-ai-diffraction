# Zenodo Assets

Current draft deposition:

- https://zenodo.org/deposit/19558452

Primary archive:

- `zenodo_paper_repro.tar.gz`

This archive contains:

- selected paper checkpoints
- compact result JSONs
- paper configs
- launchers
- short reproducibility notes

Expected local placement for downloaded checkpoints:

- `external/checkpoints/`

Use:

- [checkpoint_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/checkpoint_manifest.csv)
- [dataset_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/dataset_manifest.csv)

as the mapping from paper roles to:

- Zenodo checkpoint filenames
- expected local checkpoint paths
- required external datasets and environment variables

When the final paper-only GitHub repo exists, update this file with:

1. the final Zenodo record URL
2. the GitHub repo URL
3. any related DOI links
4. a short mapping from Zenodo archive contents to repo paths
