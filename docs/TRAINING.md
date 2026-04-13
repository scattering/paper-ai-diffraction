# Training

This repository is designed so that the paper code lives here while the checkpoint binaries live in Zenodo.

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

- [checkpoint_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/checkpoint_manifest.csv)
- [zenodo_files.md](/tmp/paper-ai-diffraction/reproducibility/zenodo_files.md)

## External Data

The original paper campaign used external HDF5 datasets that are not bundled here. In a final paper repo, these should be documented in a dataset manifest with stable names.

The main expected dataset roles are:

- standard RRUFF-conditioned trainready dataset
- PO trainready dataset
- `RRUFF-325` benchmark HDF5
- `RRUFF-473` benchmark HDF5
- prior/train HDF5 for Bayesian calibration priors

## Included Training Configs

- [final_mixed_2500k_dualsource.json](/tmp/paper-ai-diffraction/configs/final_mixed_2500k_dualsource.json)
- [mixed_200k_pilot.json](/tmp/paper-ai-diffraction/configs/mixed_200k_pilot.json)
- [po_1epoch.json](/tmp/paper-ai-diffraction/configs/po_1epoch.json)
- [po_resume2.json](/tmp/paper-ai-diffraction/configs/po_resume2.json)

## Included Training Launchers

These are the actual campaign launchers copied from the source project:

- [train_po_1epoch_tacc.sh](/tmp/paper-ai-diffraction/scripts/train_po_1epoch_tacc.sh)
- [train_po_resume2_tacc.sh](/tmp/paper-ai-diffraction/scripts/train_po_resume2_tacc.sh)
- [train_mixed_200k_pilot_tacc.sh](/tmp/paper-ai-diffraction/scripts/train_mixed_200k_pilot_tacc.sh)
- [train_final_mixed_2500k_dualsource_tacc.sh](/tmp/paper-ai-diffraction/scripts/train_final_mixed_2500k_dualsource_tacc.sh)

These are useful as faithful records of how the campaign was run, but they still contain TACC-specific assumptions and should be treated as historical launchers rather than polished general-purpose entrypoints.

## Paper-Relevant Training Lineage

- `ic6gfmvm`
  - Stage-1 base checkpoint
- `9rwv1qly`
  - Stage-2c no-PO baseline
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
