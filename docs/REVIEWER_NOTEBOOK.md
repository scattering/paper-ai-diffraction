# Reviewer Notebook

This repo now includes a reviewer-facing notebook:

- `notebooks/reviewer_walkthrough.ipynb`

The notebook is designed around single-pattern CSV inference rather than benchmark HDF5 ingestion.
It also knows how to browse a precomputed paper-backed `RRUFF-325` summary JSON so reviewers do not need to rerun benchmark inference inside Jupyter.

Reviewers do **not** need Box, TACC, or the full RRUFF benchmark to use the shipped notebook examples and bundled compact reviewer assets.

## Input Contract

For an arbitrary reviewer-supplied pattern, provide a CSV with:

- `2theta` or `two_theta`
- `intensity`

The notebook:

- sorts the pattern by `2theta`
- normalizes intensity internally
- linearly interpolates onto the model grid
- runs extinction-group inference from a released checkpoint

Reviewers do **not** need to pre-match the model's `2theta` step.

## Included Real Examples

The repo also includes two real benchmark-derived examples:

- `assets/reviewer_examples/correct_case_015_*.csv`
- `assets/reviewer_examples/failure_case_000_*.csv`

These were exported from the local `RRUFF-325` benchmark copy using the final mixed paper model's per-example outcome JSON.

Each CSV has a paired metadata JSON with:

- benchmark index
- case ID
- mineral name
- true extinction group
- the paper model's stored top-5 prediction summary

In addition, if `results/reviewer/rruff325_precomputed_inference.json` is present, the notebook can inspect the full 325-example summary directly. That summary is compact, paper-backed, and intended for TAP/Zenodo distribution.

The public paper repo intentionally does **not** redistribute the full RRUFF benchmark HDF5s. Instead it ships:

- two benchmark-derived example CSVs
- compact JSON metadata for those examples
- derived SG/EG mapping CSVs
- compact prior JSON/CSV
- a compact precomputed benchmark summary JSON

## Optional Calibrated Inference

The notebook supports two inference modes:

1. checkpoint-only inference
   This works with just a released checkpoint and config.

2. paper-faithful calibrated inference
   This additionally uses:
   - `PRIOR_H5`
   - temperature scaling

This distinction matters because the paper's final deployed inference uses the temperature-scaled auxiliary head plus a training-derived empirical prior.

## TACC-Side Artifact Generators

To reduce reviewer friction, this repo now includes three compact artifact generators:

- `scripts/export_prior_asset.py`
- `scripts/export_rruff_examples.py`
- `scripts/precompute_benchmark_inference.py`

Recommended TACC workflow:

1. export a compact prior JSON/CSV from the training HDF5
2. export one correct and one failure example as plain CSV
3. optionally precompute benchmark-wide per-example summaries for browsing

Those artifacts are small and are the right candidates for later Zenodo inclusion.
