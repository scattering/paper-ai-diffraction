# Zenodo Assets

Current Zenodo draft:
- [zenodo.org/deposit/19558452](https://zenodo.org/deposit/19558452)

Final paper repo:
- [github.com/scattering/paper-ai-diffraction](https://github.com/scattering/paper-ai-diffraction)

Primary archive:
- `zenodo_paper_repro.tar.gz`

Archive contents:
- selected paper checkpoints
- compact result JSONs
- paper configs
- canonical and archived launchers
- short reproducibility notes

Expected local placement for downloaded checkpoints:

```text
external/checkpoints/
```

Expected workflow:
1. Download the Zenodo archive or selected checkpoint files.
2. Place the checkpoint files under `external/checkpoints/`.
3. Use [checkpoint_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/checkpoint_manifest.csv) to match paper roles to checkpoint filenames.
4. Use [dataset_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/dataset_manifest.csv) to provide required external benchmark and trainready datasets.

After Zenodo publication, update this file with:
1. the final DOI
2. the final record URL
3. any article DOI link
