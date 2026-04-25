# Training

This repository is designed so that the paper code lives here while the checkpoint binaries live in Zenodo.

Before using the wrappers below:

```bash
conda env create -f environment-train-eval.yml
conda activate paper-ai-diffraction-train-eval
pip install -e .
```

For TACC-specific notes, see:

- [TACC_ENV.md](TACC_ENV.md)

Expected local checkpoint placement:

```text
external/checkpoints/
  xrd_model_ic6gfmvm_best.pth
  xrd_model_9rwv1qly_best.pth
  xrd_model_cscjfdwk_best.pth
  xrd_model_dsi7ehiv_best.pth
  xrd_model_eeru8svx_best.pth
  xrd_model_82ept35h_best.pth
```

See:

- [checkpoint_manifest.csv](../reproducibility/checkpoint_manifest.csv)
- [dataset_manifest.csv](../reproducibility/dataset_manifest.csv)
- [zenodo_files.md](../reproducibility/zenodo_files.md)

## External Data

The original paper campaign used external HDF5 datasets that are not bundled here. The current prototype now records them in:

- [dataset_manifest.csv](../reproducibility/dataset_manifest.csv)

The main expected dataset roles are:

- standard RRUFF-conditioned trainready dataset
- PO trainready dataset
- `RRUFF-325` benchmark HDF5
- `RRUFF-473` benchmark HDF5
- prior/train HDF5 for Bayesian calibration priors

## Included Training Configs

- [final_mixed_2500k_dualsource.json](../configs/final_mixed_2500k_dualsource.json)
- [mixed_200k_pilot.json](../configs/mixed_200k_pilot.json)
- [po_1epoch.json](../configs/po_1epoch.json)
- [po_resume2.json](../configs/po_resume2.json)

## Canonical Training Wrappers

Use these first:

- [train_po_1epoch.sh](../scripts/train_po_1epoch.sh)
- [train_po_resume2.sh](../scripts/train_po_resume2.sh)
- [train_mixed_200k.sh](../scripts/train_mixed_200k.sh)
- [train_final_mixed.sh](../scripts/train_final_mixed.sh)

These wrappers accept data and checkpoint locations through environment variables, write a temporary config with local overrides, and then call [train.py](../src/paper_ai_diffraction/core/train.py) through the installed package without assuming a specific cluster path layout.

## Historical TACC Launchers

The original campaign launchers are preserved under:

- [../scripts/tacc_archive]( ../scripts/tacc_archive )

Use those only if you need an exact record of the April 2026 TACC batch jobs.

## Paper-Relevant Training Lineage

- `ic6gfmvm`
  - Stage-1 base checkpoint
- `9rwv1qly`
  - Stage-2c no-PO baseline
- `cscjfdwk`
  - PO-only 1-epoch ablation
- `dsi7ehiv`
  - PO continuation / effective 2-epoch comparison
- `eeru8svx`
  - mixed-200k pilot
- `82ept35h`
  - final large mixed dual-source champion

## Preferred-Orientation And Mixed-Curriculum Summary

The public repo keeps only the essential paper-facing lineage.

- Preferred orientation was implemented as exact March-Dollase reweighting in the active CFML powder-generation path.
- The PO corpus remained a dirty Stage-2-style synthetic distribution rather than a clean texture-only corpus.
- The final champion training path used a dual-source loader rather than physically merging giant HDF5 files.

The practical implication is:

- use the PO runs as ablations showing the effect of explicit texture modeling
- use the dual-source large mixed run as the recommended final balanced model

The detailed dated handoff notes remain in the private source repository, not in this public paper repo.
- `cscjfdwk`
  - PO-only 1-epoch checkpoint
- `dsi7ehiv`
  - PO continuation checkpoint
- `eeru8svx`
  - mixed-200k pilot
- `82ept35h`
  - final large mixed champion

## Practical Advice

If the goal is to reproduce the paper, the shortest path is:

1. download checkpoints from Zenodo
2. use the evaluation wrappers in `scripts/`
3. regenerate the compact paper tables and topology figures

If the goal is to replay training, use the copied configs and launchers here as the reference implementation, but expect to adapt paths and scheduler settings.
