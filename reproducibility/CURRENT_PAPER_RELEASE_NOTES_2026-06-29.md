# Current Paper Release Notes

Date: 2026-06-29

This note documents the current public GitHub artifact surface for the PRX
paper.
The paper cites this repository as:

- <https://github.com/scattering/paper-ai-diffraction>

## Current GitHub Contents

- Flat-37 mapping assets:
  - `assets/lookups/eg37_mapping_20260622.json`
  - `assets/lookups/eg37_mapping_20260622.csv`
- Flat-37 ViT configs:
  - `configs/flat37/vit_stage1_uniform_2m_20260622.json`
  - `configs/flat37/vit_stage2c_rruff2346k_20260622.json`
  - `configs/flat37/vit_po200k_20260622.json`
  - `configs/flat37/vit_dualsource2346k_500kpo_20260622.json`
- Matched SG-to-EG control configs:
  - `configs/flat37/sg230_categorical_control_20260624.json`
  - `configs/flat37/eg99_categorical_control_20260624.json`
- Matched regular-transformer configs:
  - `configs/regular_transformer/rt_stage1_uniform_2m_flat37_20260624.json`
  - `configs/regular_transformer/rt_stage2c_rruff2346k_flat37_20260624.json`
- TACC provenance wrappers under:
  - `scripts/tacc_archive/flat37/`
  - `scripts/tacc_archive/regular_transformer/`
- Grouped mineral-family calibration assets under:
  - `results/flat37/calibration/`
- Information-channel and topology summaries under:
  - `results/flat37/information_theory/`
- Summary table source:
  - `results/revised_paper_summary.json`

The utility that builds flat-37 targets derives each space group's canonical
row from `canonical_extinction_to_space_group.csv` membership and checks the
lookup-table symbol for consistency.

## Current Paper Reference Numbers

Stage-2c flat-37 ViT, calibrated Bayesian auxiliary output at `T=5`:

- RRUFF-325: Top-1/3/5 = 10.77 / 26.15 / 42.46
- RRUFF-473: Top-1/5 = 11.84 / 49.68

Large mixed flat-37 ViT:

- RRUFF-325: Top-1/5 = 11.69 / 43.69
- RRUFF-473: Top-1/5 = 12.47 / 51.37

Matched regular transformer:

- RRUFF-325: Top-1/3/5 = 10.77 / 30.15 / 43.69
- RRUFF-473: Top-1/3/5 = 12.68 / 34.46 / 51.16

Matched SG-to-EG control:

- raw SG space: 8.54 / 17.89 / 24.15
- SG probabilities collapsed to EG space: 10.22 / 21.67 / 29.30
- direct EG: 12.05 / 24.17 / 31.96

Numbers are Top-1/3/5 unless otherwise stated.

## Zenodo/Public-Repo Release State

The current Zenodo package for this release is published as:

- DOI: [10.5281/zenodo.21048093](https://doi.org/10.5281/zenodo.21048093)
- Record: [zenodo.org/records/21048093](https://zenodo.org/records/21048093)
- Archive: `paper_ai_diffraction_revised_20260629.tar.gz`

- It includes the final SG-to-EG JSON/CSV outputs:
  - `/scratch/09870/williamratcliff/sg_eg_categorical_20260624/`
- It includes the final regular-transformer evaluation JSONs:
  - `/scratch/09870/williamratcliff/rt_flat37_20260624/784439/`
- It includes the current Stage-1, PO-only, and large mixed flat-37 checkpoints
  named in `checkpoint_manifest.csv`.
- It includes the matched regular-transformer checkpoint with its configs
  and compact evaluation JSONs, as the architecture-control comparison.
- It includes `benchmark_ids/rruff325_case_ids.csv` and
  `benchmark_ids/rruff473_case_ids.csv`.
- Do not include posterior `.npz` files in the main Zenodo package. Compact JSON
  summaries are sufficient for table reproduction; exact posterior audit replay
  would be a separate audit-companion package.
- Reviewer notebook examples and lightweight browsing assets are tracked in the
  GitHub repo with the notebook support files.

WiSE-FT is not part of the current-paper final model package. If kept in the
archive, it should be labeled historical or exploratory.
