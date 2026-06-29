# paper-ai-diffraction

Paper-focused reproducibility repository for *Attention Is Not All You Need for Diffraction*.

This repo reproduces the current paper-facing table rows and figure layer for the diffraction-symmetry study:
- benchmark summary rows from bundled JSON artifacts
- flat-37 ViT configuration and mapping assets
- matched regular-transformer and SG-to-EG control configuration/provenance files
- topology, calibration, and information-channel summary artifacts

The repo does **not** bundle model checkpoints or benchmark HDF5 files. Those come from:
- Zenodo checkpoints: see [reproducibility/checkpoint_manifest.csv](reproducibility/checkpoint_manifest.csv)
- external benchmark/trainready datasets: see [reproducibility/dataset_manifest.csv](reproducibility/dataset_manifest.csv)

Because the upstream RRUFF-derived benchmark files are not redistributed here, this repo
publishes the benchmark-construction algorithms instead:
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- [scripts/reconstruct_rruff_473.py](scripts/reconstruct_rruff_473.py)
- [scripts/build_rruff_325_from_473.py](scripts/build_rruff_325_from_473.py)

The repo *does* bundle compact reviewer-facing and paper-audit artifacts:
- two example diffraction CSVs derived from the paper benchmark
- their paired JSON metadata
- SG/EG lookup CSVs
- the 99-EG to flat-37 rule-output mapping
- a compact prior JSON/CSV
- a compact precomputed `RRUFF-325` summary JSON

Those are sufficient for the shipped notebook walkthrough without Box or the full RRUFF benchmark.

Reviewer-facing notebook support is documented in:
- [REVIEWER_NOTEBOOK.md](docs/REVIEWER_NOTEBOOK.md)
- [SUPPLEMENTAL.md](docs/SUPPLEMENTAL.md)

Supported notebook usage paths:
- local machine with the train/eval environment and a released checkpoint
- TACC TAP on Stampede3 with the same repo checkout and checkpoint placement

Google Colab is plausible for the lightweight checkpoint-only reviewer demo, but it is not the primary validated path.

Zenodo archival package:
- DOI: [10.5281/zenodo.21043734](https://doi.org/10.5281/zenodo.21043734)
- Record: [zenodo.org/records/21043734](https://zenodo.org/records/21043734)

Current archive split:
- GitHub repo:
  - code
  - notebooks
  - benchmark-construction scripts
  - paper-facing wrappers and docs
- Zenodo:
  - checkpoints
  - compact result JSONs
  - configs
  - launchers
  - short archival manifests/notes
- reviewer compact assets:
  - tracked directly in this GitHub repo with the reviewer notebook support files

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
external/checkpoints/                         # core paper checkpoints
external/checkpoints/                         # supplemental ViT/RT/CNN checkpoints
```

The exact filenames and expected local paths are listed in:
- [reproducibility/checkpoint_manifest.csv](reproducibility/checkpoint_manifest.csv)

This manifest includes every checkpoint explicitly named in the manuscript main text or supplement. Exploratory checkpoints are still archived, but are labeled accordingly in the notes column.

External benchmark and trainready datasets are not redistributed in this repo. Their required environment variables and example source paths are listed in:
- [reproducibility/dataset_manifest.csv](reproducibility/dataset_manifest.csv)

## Regenerate Paper Outputs

Table rows from bundled paper JSONs:

```bash
python scripts/make_main_tables.py
```

The default table script reads [results/revised_paper_summary.json](results/revised_paper_summary.json), which carries the revised flat-37 ViT, regular-transformer, and matched SG-to-EG control numbers.

Topology-distance figure from bundled failure JSONs:

```bash
./scripts/make_topology_distance_figure.sh
```

Topology-flow figure set from bundled failure JSON plus external canonical CSV:

```bash
export CANONICAL_CSV=/path/to/canonical_extinction_to_space_group.csv
./scripts/make_topology_flow_figure.sh
```

Calibration sweep figure from the bundled Stage-2c sweep JSON:

```bash
./scripts/make_calibration_figure.sh
```

Curriculum holdout figure from bundled paper values:

```bash
python scripts/make_curriculum_real_holdout.py
```

RRUFF-473 decoder-tradeoff figure from bundled paper values:

```bash
python scripts/make_stage_decoder_tradeoffs_rruff473.py
```

Physics-PE supplementary ruler figure from the bundled checkpoint-curve JSON:

```bash
python scripts/make_physics_pe_q2_ruler.py
```

Reconstruct the frozen `RRUFF-473` benchmark from an upstream manifest plus raw XY files:

```bash
python scripts/reconstruct_rruff_473.py --manifest-json /path/to/rruff_cukalpha_manifest.json --xy-dir /path/to/xy_raw --reference-manifest-json /path/to/option1_metadata_manifest.json --output-json results/rruff473_reconstruction_summary.json
```

Rebuild `RRUFF-325` deterministically from frozen `RRUFF-473`:

```bash
python scripts/build_rruff_325_from_473.py --input-h5 /path/to/RRUFF_option1_473_with_buckets_maxnorm.hdf5 --output-h5 /path/to/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5
```

Reviewer-support artifact generation:

```bash
python scripts/export_prior_asset.py --prior-h5 /path/to/trainready.hdf5 --output-csv results/reviewer/ext_group_priors.csv --output-json results/reviewer/ext_group_priors.json
python scripts/export_rruff_examples.py --benchmark-h5 /path/to/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 --failure-json results/mixed2500k_compare_325_failure_modes_655279.json --output-dir assets/reviewer_examples
python scripts/precompute_benchmark_inference.py --checkpoint external/checkpoints/xrd_model_pubfix_flat37_20260622_stage2c_r2346k_s1337_best.pth --config configs/flat37/vit_stage2c_rruff2346k_20260622.json --benchmark-h5 /path/to/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 --prior-h5 /path/to/trainready.hdf5 --output-json results/reviewer/rruff325_precomputed_inference.json
```

If `results/reviewer/rruff325_precomputed_inference.json` is present, the reviewer notebook can browse the full paper-backed 325-example summary directly instead of recomputing it inside Jupyter.

## Repo Contract

- `results/` contains only compact paper-backed JSON artifacts.
- `results/figures/` is generated output and is not tracked.
- `scripts/` contains the canonical paper-facing wrappers.
- `scripts/tacc_archive/` contains preserved historical campaign launchers for provenance only.
- [reproducibility/REVISED_PAPER_RELEASE_NOTES_2026-06-29.md](reproducibility/REVISED_PAPER_RELEASE_NOTES_2026-06-29.md) records the current paper assets and the remaining Zenodo/public-repo staging items.

## Key References

- [reproducibility/checkpoint_manifest.csv](reproducibility/checkpoint_manifest.csv)
- [reproducibility/dataset_manifest.csv](reproducibility/dataset_manifest.csv)
- [reproducibility/zenodo_files.md](reproducibility/zenodo_files.md)
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- [docs/EVALUATION.md](docs/EVALUATION.md)
- [docs/FIGURES.md](docs/FIGURES.md)
