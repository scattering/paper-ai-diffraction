# Student Handoff: Preferred Orientation, Dual-Source Mixed Training, and Final Champion Model

## Scope

This note records the exact preferred-orientation (PO) implementation, the training/evaluation lineage that followed, and the current recommended final model.

## Main conclusion

The current best **balanced** model is the large dual-source mixed run:

- training job: `655279`
- checkpoint: `/scratch/09870/williamratcliff/ai_diffraction_models/xrd_model_82ept35h_best.pth`
- config: `/scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/config_rruff_conditioned_dualsource_2346k_500kpo_from_ic6gfmvm_physpe_coord.json`
- metadata: `/scratch/09870/williamratcliff/dualsource2500k_train_655279.json`

Headline real-data results:

- `RRUFF-325`: calibrated Bayesian aux `Top-1 / Top-5 = 10.46% / 43.69%`
- `RRUFF-473`: calibrated Bayesian aux `Top-1 / Top-5 = 9.94% / 50.74%`
- split-head validity:
  - `RRUFF-325`: `47.38%`
  - `RRUFF-473`: `49.26%`
- DAG on `RRUFF-325`:
  - descendant / ancestor / branch = `165 / 31 / 95`
  - `% <= 2 hops = 51.55%`
  - mean DAG distance = `2.46`

Interpretation:

- the pure-PO model gives the sharpest Top-1
- the large mixed dual-source model gives the best overall balance of:
  - Top-5 breadth
  - Top-1 improvement over Stage-2c
  - split-head self-consistency
  - topological locality

## What changed technically

### 1. Exact March-Dollase preferred orientation

Preferred orientation was implemented as exact March-Dollase reweighting inside the active CFML Fortran wrapper used by our current `CFML_api` powder-generation route.

This is **not** a post hoc profile warp.

Validated facts:

- changing only `po_r` changes the generated pattern materially
- a reflection-family sanity check with preferred direction `(001)` confirmed the expected family-selective enhancement/suppression
- PO is physically active in the synthetic generator

### 2. PO generation still includes ordinary Stage-2 nuisance structure

The PO corpus remains a dirty geological-style synthetic corpus.

The generator still includes:

- impurity generation/mixing
- empirical backgrounds
- existing noise/line-shape paths

PO is an added nuisance axis, not a replacement for ordinary Stage-2 dirtiness.

## Datasets

### PO datasets

- `200k` PO trainready:
  - `/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_200k_po_v1_trainready.hdf5`
- `300k` PO trainready:
  - `/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_300k_po_v1_trainready.hdf5`
- merged `500k` PO trainready:
  - `/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5`

### Standard Stage-2 dataset

- `/work2/09870/williamratcliff/rruff_conditioned_2346k_v1_trainready.hdf5`
- Vista scratch copy used for the successful dual-source run:
  - `/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5`

## Why the giant merged-HDF5 path is no longer preferred

We attempted to build a physical `2.5M` merged trainready HDF5 on Stampede.

That path repeatedly timed out:

- `skx-dev` build hit the `2h` walltime
- `spr` build also timed out at `6h`

Root cause was not a Python exception; the operation is just an expensive compressed HDF5 reshuffle over very large arrays.

Recommended lesson:

- **do not** rebuild giant mixed HDF5 files for every new ratio
- instead, train from multiple source HDF5s with a mixed sampler / mixed dataset mode

## Dual-source mixed loader

We implemented a no-merge mixed-HDF5 training mode in the ViT code.

Key code paths:

- `/Users/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/dataset.py`
- `/Users/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/train.py`

Concept:

- keep standard and PO HDF5 files separate
- sample from both during training
- avoid the costly one-time physical merge

Advantages:

- no giant repack step
- easier ratio changes
- easier future ablations
- lower storage duplication

## Training lineage

All late runs below start from the same uniform base:

- base checkpoint: `ic6gfmvm`

### PO-only, 1 epoch

- run id: `cscjfdwk`
- `RRUFF-325`: `13.54 / 40.92`
- `RRUFF-473`: `13.53 / 49.89`
- split-head validity on `RRUFF-325`: `0.92%`
- DAG on `RRUFF-325`: `157 / 28 / 96`, `53.0% <=2`, mean `2.51`

This is still the sharpest Top-1 result.

### PO-only continuation to 2 effective epochs

- run id: `dsi7ehiv`
- synthetic held-out improves slightly
- real Top-1 slips relative to 1-epoch PO

Conclusion:

- the useful PO effect is early
- continuing longer did not improve real-data Top-1

### Mixed `200k` pilot

- run id: `eeru8svx`
- `RRUFF-325`: `13.23 / 40.92`
- `RRUFF-473`: `13.53 / 49.05`
- split-head validity on `RRUFF-325`: `1.54%`
- DAG on `RRUFF-325`: `153 / 30 / 99`, `49.65% <=2`, mean `2.59`

Interpretation:

- good compromise
- not a clear win over the 1-epoch PO model
- did not restore broad Top-5 enough

### Final large dual-source mixed run

- job: `655279`
- run id: `82ept35h`
- mode: dual-source mixed loader
- training length: exactly `1` epoch
- standard source on Vista scratch: `2346k`
- PO source on Vista scratch: `500k`

Real-data eval job:

- `655900` on `gh-dev`

Outputs:

- `/scratch/09870/williamratcliff/mixed2500k_calibration_metrics_325_655279.json`
- `/scratch/09870/williamratcliff/mixed2500k_calibration_metrics_473_655279.json`
- `/scratch/09870/williamratcliff/mixed2500k_split_validity_325_655279.json`
- `/scratch/09870/williamratcliff/mixed2500k_split_validity_473_655279.json`
- `/scratch/09870/williamratcliff/mixed2500k_compare_325_failure_modes_655279.json`

## Current scientific interpretation

### What PO taught us

- explicit preferred orientation matters
- pure-PO training sharpens exact Top-1
- it also weakens descendant bias and makes errors more local

### What pure PO does not give us

- broad Top-5 coverage on `RRUFF-325`
- stable split-head exact decoding on real mixtures

### What the large mixed model gives us

- restores broad Top-5 (`43.69%` on `RRUFF-325`)
- modestly improves Top-1 over Stage-2c (`10.46%` vs `9.54%`)
- restores split-head validity (`47.38%` on `RRUFF-325`)
- gives the tightest mean DAG distance (`2.46`)

### Current topology interpretation

The best current interpretation is local “texture aliasing” / cousin-confusion:

- preferred orientation suppresses whole reflection families
- once those are weakened, the remaining 1D trace can resemble a nearby crystallographic cousin
- branch jumps remain elevated, but they stay very local in graph space

So the final large mixed model is the best **balanced** deployment model, while the pure-PO run remains the best ablation for showing that subtractive nuisance physics matters.

## Recommended next steps

1. Use the final large mixed dual-source model as the paper’s balanced champion.
2. Keep the pure-PO 1-epoch result as the mechanistic ablation showing the value of explicit preferred orientation.
3. If doing more training, prefer dual-source or multi-source loading over physical mega-HDF5 merges.
4. If doing more eval, try:
   - broader full-RRUFF robustness evaluation
   - temperature sweep around the final large mixed run
   - additional DAG comparisons on broader real sets

## Key manuscript / note files

- manuscript working note:
  - `/Users/williamratcliff/Library/CloudStorage/Box-Box/tasai_paper_clean/preferred_orientation_status_2026-04-04.md`
- repo note copy:
  - `/Users/williamratcliff/ai-diffraction/docs/review_notes/preferred_orientation_status_2026-04-04.md`
- paper draft updated with these results:
  - `/Users/williamratcliff/Downloads/paper_prx_intelligence.tex`
