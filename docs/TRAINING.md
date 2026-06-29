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
  xrd_model_pubfix_flat37_20260622_stage2c_r2346k_s1337_best.pth
  xrd_model_pubfix_rt_flat37_20260624_stage2c_r2346k_s1337_best.pth
  # additional supplemental and historical checkpoints listed in the manifests
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

- [flat37/vit_stage1_uniform_2m_20260622.json](../configs/flat37/vit_stage1_uniform_2m_20260622.json)
- [flat37/vit_stage2c_rruff2346k_20260622.json](../configs/flat37/vit_stage2c_rruff2346k_20260622.json)
- [flat37/vit_po200k_20260622.json](../configs/flat37/vit_po200k_20260622.json)
- [flat37/vit_dualsource2346k_500kpo_20260622.json](../configs/flat37/vit_dualsource2346k_500kpo_20260622.json)
- [regular_transformer/rt_stage1_uniform_2m_flat37_20260624.json](../configs/regular_transformer/rt_stage1_uniform_2m_flat37_20260624.json)
- [regular_transformer/rt_stage2c_rruff2346k_flat37_20260624.json](../configs/regular_transformer/rt_stage2c_rruff2346k_flat37_20260624.json)

The older mixed-200k and `82ept35h` configs remain in `configs/` for provenance.

## Canonical Training Wrappers

Use these first for the revised flat-37 campaign:

- [scripts/tacc_archive/flat37/vista_train_flat37_publication_20260622.sh](../scripts/tacc_archive/flat37/vista_train_flat37_publication_20260622.sh)
- [scripts/tacc_archive/flat37/submit_flat37_publication_reruns_20260622.sh](../scripts/tacc_archive/flat37/submit_flat37_publication_reruns_20260622.sh)
- [scripts/tacc_archive/regular_transformer/vista_train_rt_flat37_stage1_20260624.sh](../scripts/tacc_archive/regular_transformer/vista_train_rt_flat37_stage1_20260624.sh)
- [scripts/tacc_archive/regular_transformer/vista_train_rt_flat37_stage2c_epoch_chunk_ddp_20260624.sh](../scripts/tacc_archive/regular_transformer/vista_train_rt_flat37_stage2c_epoch_chunk_ddp_20260624.sh)

These wrappers are preserved as TACC provenance for the publication runs. The portable wrappers in the top-level `scripts/` directory remain useful for local adaptation, but the revised-paper campaign was run through the TACC wrappers above.

## Historical TACC Launchers

The original and revision campaign launchers are preserved under:

- [../scripts/tacc_archive]( ../scripts/tacc_archive )

Use those only if you need an exact record of the TACC batch jobs.

## Paper-Relevant Training Lineage

- `xrd_model_pubfix_flat37_20260622_stage2c_r2346k_s1337_best.pth`
  - revised non-PO real-benchmark reference
- flat-37 PO-only and large mixed checkpoints
  - revised preferred-orientation and final mixed-curriculum variants; exact Zenodo filenames remain to be confirmed in the release manifest
- `xrd_model_pubfix_rt_flat37_20260624_stage2c_r2346k_s1337_best.pth`
  - matched regular-transformer architecture control
- `9rwv1qly`, `cscjfdwk`, `dsi7ehiv`, `eeru8svx`, `82ept35h`
  - historical archive checkpoints retained for provenance

## Preferred-Orientation And Mixed-Curriculum Summary

The public repo keeps only the essential paper-facing lineage.

- Preferred orientation was implemented as exact March-Dollase reweighting in the active CFML powder-generation path.
- The PO corpus remained a dirty Stage-2-style synthetic distribution rather than a clean texture-only corpus.
- The final champion training path used a dual-source loader rather than physically merging giant HDF5 files.

The practical implication is:

- use the PO runs as ablations showing the effect of explicit texture modeling
- use the dual-source large mixed flat-37 run as the recommended final balanced model

The detailed dated handoff notes remain in the private source repository, not in this public paper repo.

## Practical Advice

If the goal is to reproduce the paper, the shortest path is:

1. download checkpoints from Zenodo
2. use the evaluation wrappers in `scripts/`
3. regenerate the compact paper tables and topology figures

If the goal is to replay training, use the copied configs and launchers here as the reference implementation, but expect to adapt paths and scheduler settings.
