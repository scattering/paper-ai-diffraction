# paper-ai-diffraction

Paper-focused reproducibility repository for *Attention Is Not All You Need for Diffraction*.

This repo reproduces the paper-facing table rows and figure layer for the mixed-curriculum results:
- benchmark summary rows from bundled JSON artifacts
- topology-distance figure
- topology-flow figure set

The repo does **not** bundle model checkpoints or benchmark HDF5 files. Those come from:
- Zenodo checkpoints: see [reproducibility/checkpoint_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/checkpoint_manifest.csv)
- external benchmark/trainready datasets: see [reproducibility/dataset_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/dataset_manifest.csv)

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
- [TACC_ENV.md](/tmp/paper-ai-diffraction/docs/TACC_ENV.md)

## Checkpoints And Data

Checkpoints are downloaded from Zenodo and should be placed under:

```text
external/checkpoints/
```

The exact filenames expected by the wrappers are listed in:
- [reproducibility/checkpoint_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/checkpoint_manifest.csv)

External benchmark and trainready datasets are not redistributed in this repo. Their required environment variables and example source paths are listed in:
- [reproducibility/dataset_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/dataset_manifest.csv)

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

## Repo Contract

- `results/` contains only compact paper-backed JSON artifacts.
- `results/figures/` is generated output and is not tracked.
- `scripts/` contains the canonical paper-facing wrappers.
- `scripts/tacc_archive/` contains preserved historical campaign launchers for provenance only.

## Key References

- [reproducibility/checkpoint_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/checkpoint_manifest.csv)
- [reproducibility/dataset_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/dataset_manifest.csv)
- [reproducibility/zenodo_files.md](/tmp/paper-ai-diffraction/reproducibility/zenodo_files.md)
- [docs/EVALUATION.md](/tmp/paper-ai-diffraction/docs/EVALUATION.md)
- [docs/FIGURES.md](/tmp/paper-ai-diffraction/docs/FIGURES.md)
