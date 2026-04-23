# Provenance Notebooks

This repo keeps a small number of notebooks as archival provenance artifacts when they help explain how a paper figure or supplemental artifact was generated, but do not rise to the level of a stable paper-facing script.

Current provenance notebooks:

- [notebooks/provenance/wandb_confusion_matrix_from_table.ipynb](/tmp/paper-ai-diffraction-fresh/notebooks/provenance/wandb_confusion_matrix_from_table.ipynb)
  - original source: Derrick Chan-Sew
  - purpose: reconstruct the supplemental ViT confusion matrix from a Weights & Biases confusion-matrix table artifact
  - expected input: a W&B table JSON such as `media_table_conf_mat_top1_inference_xrd_model_duninwhx_table_4_89d4c914203af59fa295.table.json`
  - limitation: exact rerun still requires the correct model-to-artifact mapping for the paper figure

These notebooks are retained for transparency and convenience. They should not be treated as the primary paper-facing reproduction path unless that workflow is also documented in `README.md` or `docs/FIGURES.md`.
