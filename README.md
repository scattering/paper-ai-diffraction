# paper-ai-diffraction

Paper-focused reproducibility repository for *Attention Is Not All You Need for Diffraction*.

This repo reproduces the paper-facing table rows and figure layer for the mixed-curriculum results:
- benchmark summary rows from bundled JSON artifacts
- supplemental positional-ablation benchmark rows from bundled JSON artifacts
- topology-distance figure
- topology-flow figure set

The repo does **not** bundle model checkpoints or benchmark HDF5 files. Those come from:
- Zenodo checkpoints: see [reproducibility/checkpoint_manifest.csv](reproducibility/checkpoint_manifest.csv)
- external benchmark/trainready datasets: see [reproducibility/dataset_manifest.csv](reproducibility/dataset_manifest.csv)

The repo *does* bundle compact reviewer-facing artifacts:
- two example diffraction CSVs derived from the paper benchmark
- their paired JSON metadata
- SG/EG lookup CSVs
- a compact prior JSON/CSV
- a compact precomputed `RRUFF-325` summary JSON

Those are sufficient for the shipped notebook walkthrough without Box or the full RRUFF benchmark.

Reviewer-facing notebook support is documented in:
- [REVIEWER_NOTEBOOK.md](docs/REVIEWER_NOTEBOOK.md)

Supported notebook usage paths:
- local machine with the train/eval environment and a released checkpoint
- TACC TAP on Stampede3 with the same repo checkout and checkpoint placement

Google Colab is plausible for the lightweight checkpoint-only reviewer demo, but it is not the primary validated path.

Current Zenodo draft:
- [zenodo.org/deposit/19558452](https://zenodo.org/deposit/19558452)

After publication, replace the draft link above with the final Zenodo DOI and record URL.

## Install

For paper tables and figures:

```bash
conda env create -f environment.yml
conda activate paper-ai-diffraction
pip install -e .
```

For checkpoint evaluation or training reruns:

```bash
conda env create -f environment-train-eval.yml
conda activate paper-ai-diffraction-train-eval
pip install -e .
```

TACC-specific notes are in:
- [TACC_ENV.md](docs/TACC_ENV.md)

## Checkpoints And Data

Checkpoints are downloaded from Zenodo and should be placed under:

```text
external/checkpoints/
```

The exact filenames expected by the wrappers are listed in:
- [reproducibility/checkpoint_manifest.csv](reproducibility/checkpoint_manifest.csv)

External benchmark and trainready datasets are not redistributed in this repo. Their required environment variables and example source paths are listed in:
- [reproducibility/dataset_manifest.csv](reproducibility/dataset_manifest.csv)

## Regenerate Paper Outputs

Table rows from bundled paper JSONs:

```bash
python scripts/make_main_tables.py
```

Topology-distance figure from bundled failure JSONs:

```bash
./scripts/make_topology_distance_figure.sh
```

Topology-flow figure set from bundled failure JSON plus external canonical CSV:

```bash
export CANONICAL_CSV=/path/to/canonical_extinction_to_space_group.csv
./scripts/make_topology_flow_figure.sh
```

Calibration sweep figure from an external sweep JSON:

```bash
export CAL_SWEEP_JSON=/path/to/decoder_temp_sweep.json
./scripts/make_calibration_figure.sh
```

Reviewer-support artifact generation:

```bash
python scripts/export_prior_asset.py --prior-h5 /path/to/trainready.hdf5 --output-csv results/reviewer/ext_group_priors.csv --output-json results/reviewer/ext_group_priors.json
python scripts/export_rruff_examples.py --benchmark-h5 /path/to/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 --failure-json results/mixed2500k_compare_325_failure_modes_655279.json --output-dir assets/reviewer_examples
python scripts/precompute_benchmark_inference.py --checkpoint external/checkpoints/xrd_model_82ept35h_best.pth --config configs/final_mixed_2500k_dualsource.json --benchmark-h5 /path/to/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 --prior-h5 /path/to/trainready.hdf5 --output-json results/reviewer/rruff325_precomputed_inference.json
```

If `results/reviewer/rruff325_precomputed_inference.json` is present, the reviewer notebook can browse the full paper-backed 325-example summary directly instead of recomputing it inside Jupyter.

## Repo Contract

- `results/` contains only compact paper-backed JSON artifacts.
- `results/figures/` is generated output and is not tracked.
- `scripts/` contains the canonical paper-facing wrappers.
- `scripts/tacc_archive/` contains preserved historical campaign launchers for provenance only.

## Key References

- [reproducibility/checkpoint_manifest.csv](reproducibility/checkpoint_manifest.csv)
- [reproducibility/dataset_manifest.csv](reproducibility/dataset_manifest.csv)
- [reproducibility/zenodo_files.md](reproducibility/zenodo_files.md)
- [docs/EVALUATION.md](docs/EVALUATION.md)
- [docs/FIGURES.md](docs/FIGURES.md)
