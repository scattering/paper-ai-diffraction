# Benchmarks

This repository does not redistribute the upstream RRUFF-derived benchmark files.
It instead publishes the manuscript-facing construction procedures for the two frozen
evaluation sets used in the papers:

- `RRUFF-473`: the broader real-data benchmark
- `RRUFF-325`: the deterministic downstream subset used for calibration and topology

## What Is Released

- the algorithm for reconstructing `RRUFF-473` from an upstream Cu-like manifest plus raw XY files:
  - [scripts/reconstruct_rruff_473.py](/tmp/paper-ai-diffraction-fresh/scripts/reconstruct_rruff_473.py)
- the deterministic `RRUFF-473 -> RRUFF-325` builder:
  - [scripts/build_rruff_325_from_473.py](/tmp/paper-ai-diffraction-fresh/scripts/build_rruff_325_from_473.py)
- the expected external file names and example source paths:
  - [reproducibility/dataset_manifest.csv](/tmp/paper-ai-diffraction-fresh/reproducibility/dataset_manifest.csv)

## What Is Not Released

- the full RRUFF-derived HDF5 benchmark files
- the upstream Cu-like manifest itself
- raw RRUFF source data

Those inputs remain external because the public package is intended to release the
paper-backed algorithmic construction workflow rather than redistribute RRUFF content.

## RRUFF-473

The released reconstruction script follows the final benchmark rule used in the papers:

1. Start from an upstream Cu-like manifest plus raw XY files.
2. Restrict to the retained mineral-family set defined by the frozen released benchmark.
3. Split scans by `mineral + space group`.
4. Cluster within each group by tight lattice compatibility.
5. Admit singleton clusters only when they remain similar to the main family cluster in both lattice parameters and pattern shape.
6. Resolve residual partial clusters primarily by within-cluster medoid coherence, using similarity to the main family cluster as a secondary tie-break when those rankings disagree strongly.

The current frozen rule reproduces the released `RRUFF-473` benchmark exactly when run
against the corrected upstream manifest used in the final paper workflow.

## RRUFF-325

`RRUFF-325` is not an independent curation pass. It is a deterministic downstream slice of
`RRUFF-473` obtained by recomputing nuisance-fit severity from fixed `Rwp` thresholds and
retaining only:

- `usable_or_better`
- `recoverable`

The default thresholds in the public builder are:

- `usable_or_better`: `Rwp <= 0.12`
- `recoverable`: `0.12 < Rwp <= 0.22`
- `poor/catastrophic` split: `Rwp = 0.50`

## Typical Usage

Reconstruct `RRUFF-473`:

```bash
python scripts/reconstruct_rruff_473.py \
  --manifest-json /path/to/rruff_cukalpha_manifest.json \
  --xy-dir /path/to/xy_raw \
  --reference-manifest-json /path/to/option1_metadata_manifest.json \
  --output-json results/rruff473_reconstruction_summary.json
```

Rebuild `RRUFF-325` from frozen `RRUFF-473`:

```bash
python scripts/build_rruff_325_from_473.py \
  --input-h5 /path/to/RRUFF_option1_473_with_buckets_maxnorm.hdf5 \
  --output-h5 /path/to/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5
```
