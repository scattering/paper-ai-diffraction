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
  - copied TACC launchers plus canonical evaluation / figure wrappers
- `results/`
  - compact JSON artifacts used in the paper writeup
- `docs/`
  - concise run-lineage notes
- `reproducibility/`
  - manifest and Zenodo linkage placeholders
- `assets/`
  - small figure-support assets such as the extinction-group topology graph
- top-level environment files
  - `requirements.txt` placeholder
  - `environment.yml` based on the original ViT_NVIDIA campaign environment

## What This Is Good For

- identifying the minimal paper-facing file set
- testing a cleaner repository layout
- preparing a future dedicated GitHub paper repo

## What Still Needs Work

This is not yet a polished final paper repo. In particular:

1. `scripts/` still contains copied TACC launchers rather than cleaned canonical entrypoints.
2. imports in `src/` have not been rewritten into a coherent package layout.
3. `requirements.txt` is still only a placeholder; `environment.yml` is the more realistic current environment definition.
4. some wrappers still depend on external dataset paths that are not yet bundled here.
5. the final repo should likely prune or rewrite some cluster-specific assumptions.

The migrated Python files are syntax-valid as copied, but full runtime validation still depends on creating the Conda environment described in [environment.yml](/tmp/paper-ai-diffraction/environment.yml).

## Recommended Next Steps

1. convert `src/` into a consistent importable package
2. replace remaining copied launchers with a smaller set of canonical scripts
3. add a real environment lockfile
4. point `reproducibility/` to the final Zenodo deposition
5. initialize this as a fresh GitHub repo once the file set is finalized
