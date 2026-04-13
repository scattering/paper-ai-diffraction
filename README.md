# paper-ai-diffraction

Clean paper-focused repository staging area for reproducing the main results from *Attention Is Not All You Need for Diffraction*.

This directory was assembled as a curated extraction from the broader `ai-diffraction` project. It is intentionally narrower than the full lab repository and is intended to evolve into a dedicated paper code repository.

Nothing in this directory is live-linked back to the source project. All files here were copied into `/tmp/paper-ai-diffraction`, so the original repository at `/Users/williamratcliff/ai-diffraction` remains unchanged.

## Current Contents

- `src/core/`
  - training, inference, model, and dataset loading code
- `src/eval/`
  - calibration and split-validity evaluation code
- `src/topology/`
  - topology comparison and topology-flow plotting code
- `src/utils/`
  - extinction-group multilabel utilities
- `configs/`
  - paper-relevant training configs only
- `scripts/`
  - canonical training, evaluation, and figure wrappers
- `scripts/tacc_archive/`
  - preserved historical TACC campaign launchers
- `results/`
  - compact JSON artifacts used in the paper writeup
- `docs/`
  - concise run-lineage notes
- `reproducibility/`
  - manifests for checkpoints and datasets plus Zenodo linkage
- `assets/`
  - small figure-support assets such as the extinction-group topology graph
- top-level environment files
  - `environment.yml` based on the original ViT_NVIDIA campaign environment and treated as the current environment source of truth

## What This Is Good For

- identifying the minimal paper-facing file set
- testing a cleaner repository layout
- preparing a future dedicated GitHub paper repo

## What Still Needs Work

This is not yet a polished final paper repo. In particular:

1. imports in `src/` have not been rewritten into a coherent package layout.
2. some wrappers still depend on external dataset paths that are not yet bundled here.
3. the final repo should likely prune or rewrite some cluster-specific assumptions.

The migrated Python files are syntax-valid as copied, but full runtime validation still depends on creating the Conda environment described in [environment.yml](/tmp/paper-ai-diffraction/environment.yml).

## Recommended Next Steps

1. convert `src/` into a consistent importable package
2. add a real environment lockfile if Conda export alone is not sufficient
3. point `reproducibility/` to the final Zenodo deposition
4. initialize this as a fresh GitHub repo once the file set is finalized

## Reproducibility Manifests

Use these first:

- [reproducibility/checkpoint_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/checkpoint_manifest.csv)
- [reproducibility/dataset_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/dataset_manifest.csv)
- [reproducibility/zenodo_files.md](/tmp/paper-ai-diffraction/reproducibility/zenodo_files.md)

These files separate:

- checkpoints that should be downloaded from Zenodo
- external datasets that must be provided locally
- bundled compact JSON artifacts that are already in this repo
