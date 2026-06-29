# Revised Paper Release Notes

Date: 2026-06-29

This note tracks the public GitHub working-tree update for the revised PRX paper.
The paper cites this repository as:

- <https://github.com/scattering/paper-ai-diffraction>

## Updated In This Working Tree

- Revised flat-37 mapping assets:
  - `assets/lookups/eg37_mapping_20260622.json`
  - `assets/lookups/eg37_mapping_20260622.csv`
- Revised flat-37 ViT configs:
  - `configs/flat37/vit_stage1_uniform_2m_20260622.json`
  - `configs/flat37/vit_stage2c_rruff2346k_20260622.json`
  - `configs/flat37/vit_po200k_20260622.json`
  - `configs/flat37/vit_dualsource2346k_500kpo_20260622.json`
- Matched SG-to-EG control configs:
  - `configs/flat37/sg230_categorical_referee_20260624.json`
  - `configs/flat37/eg99_categorical_referee_20260624.json`
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
- Revised summary table source:
  - `results/revised_paper_summary.json`

The utility that builds flat-37 targets now derives each space group's canonical
row from `canonical_extinction_to_space_group.csv` membership and checks the
lookup-table symbol for consistency.

## Current Revised-Paper Reference Numbers

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

## Zenodo/Public-Repo Items Still Pending

- Copy the final SG-to-EG JSON/CSV outputs from Vista:
  - `/scratch/09870/williamratcliff/sg_eg_categorical_referee_20260624/`
- Copy the final regular-transformer evaluation JSONs from Vista:
  - `/scratch/09870/williamratcliff/rt_flat37_referee_20260624/784439/`
- Confirm exact Stage-1, PO-only, and large mixed flat-37 checkpoint filenames
  before adding final Zenodo manifest rows.
- Decide whether the matched regular-transformer checkpoint belongs in Zenodo,
  or whether configs plus compact evaluation JSONs are sufficient for the paper
  comparison.
- Refresh reviewer compact assets if the notebook should browse the revised
  Stage-2c flat-37 outputs rather than the older mixed-curriculum examples.

WiSE-FT is not part of the revised-paper final model package. If kept in the
archive, it should be labeled historical or exploratory.
