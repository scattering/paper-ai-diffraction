# Supplemental

Provenance notebooks and accuracy reference numbers for the paper supplemental. Checkpoint, dataset, and asset metadata live in the manifests:
- `reproducibility/checkpoint_manifest.csv` — W&B run IDs, spec_length, and table references
- `reproducibility/dataset_manifest.csv` — training/test HDF5 paths and bundled asset locations
- `reproducibility/MANIFEST.csv` — notebook and asset provenance

---

## Provenance Notebooks

These notebooks are retained as archival provenance artifacts explaining how supplemental figures and tables were generated. They are not the primary paper-facing reproduction path.

### ViT

- [notebooks/provenance/supplemental_vit_inference.ipynb](../notebooks/provenance/supplemental_vit_inference.ipynb)
  - purpose: batch inference for any of the 11 supplemental ViT checkpoints against its matching RRUFF benchmark HDF5; reproduces top-1/3/5 accuracy numbers for Tables S3–S7
  - expected inputs: a supplemental checkpoint from `external/checkpoints/supplemental/Models/` and the corresponding RRUFF HDF5 on Stampede3 (see `reproducibility/dataset_manifest.csv`)

- [notebooks/provenance/wandb_confusion_matrix_from_table.ipynb](../notebooks/provenance/wandb_confusion_matrix_from_table.ipynb)
  - purpose: reconstruct the supplemental ViT confusion matrix (Fig S3) from the bundled W&B table artifact
  - expected input: `assets/figure_data/confusion_matrix_smqmqi14.table.json`

- [notebooks/provenance/csv_to_db_with_bravais_vit_model.ipynb](../notebooks/provenance/csv_to_db_with_bravais_vit_model.ipynb)
  - purpose: generate Fig S5 — VIT attention overlay with HKL annotations using checkpoint `pi7r8pah`
  - expected inputs: `assets/figure_data/1k_structures.csv`, `assets/figure_data/interp_metadata_clean.csv`, `external/checkpoints/supplemental/Models/xrd_model_pi7r8pah.pth`

### RT

*(No provenance notebooks yet — RT checkpoint mapping still pending.)*

### CNN

*(No provenance notebooks yet — CNN checkpoint mapping still pending.)*

---

## ViT Accuracy Reference

W&B reference numbers for the supplemental ViT checkpoints. These are raw training/inference run values, not reproduced from bundled artifacts. For crashed runs, accuracy reflects the checkpoint state at the time of crash. W&B project: `nist-berkeley-ai-diffraction/ai-diffraction`

### Table S7 — VT Training-Distribution Effect

| Model | W&B Run ID | Top-1 | Top-3 | Top-5 |
|-------|------------|-------|-------|-------|
| VT – Fig S5 Attention | `pi7r8pah` | 29.06% | — | — |

*(Full Table S7 cross-evaluation results are in the paper; accuracy on RRUFF low-bkg full test set not listed here.)*

### Table S3 — VT Bias Cross-Evaluation

| Row | Model | Inference Run | Top-1 | Top-3 | Top-5 |
|-----|-------|---------------|-------|-------|-------|
| VT – Biased Reflection (self) | `f3sdux88` | (training summary) | 78.73% | 92.79% | 96.44% |
| VT – Biased PyXtal (self) | `kd1znx23` | (training summary) | 53.78% | 76.66% | 84.45% |
| VT – Balanced Refl. on Biased Test | `mth7zg2w` | `1ilykoiw` | 87.06% | 97.93% | 99.25% |
| VT – Balanced Refl. on Balanced Test | `mth7zg2w` | (not recovered) | 44.35% | 68.18% | 78.35% |
| VT – Balanced PyXtal on Biased Test | `3zmiyil8` | `3c2ltrca` | 50.49% | 75.44% | 84.51% |
| VT – Balanced PyXtal on Balanced Test | `3zmiyil8` | (not recovered) | 22.01% | 37.68% | 47.41% |

### Tables S4/S5 — VT Ablation: Data Generation Method and 2θ Range

Accuracy on synthetic test (Table S4, reduced models only):

| Model | Top-1 | Top-3 | Top-5 |
|-------|-------|-------|-------|
| VT – Reduced Reflection | 88.52% (W&B: 88.00%) | 98.53% | 99.44% |
| VT – Reduced PyXtal | 51.46% | 75.01% | 83.87% |

Accuracy on real RRUFF data (Table S5, from W&B inference runs):

| Model | W&B Inference Run | Top-1 | Top-3 | Top-5 |
|-------|-------------------|-------|-------|-------|
| VT – Reduced Refl. on Reduced RRUFF | `njur2oso` | 7.66% | 14.13% | 20.21% |
| VT – Full Refl. on Full RRUFF | `hrhk538h` | 7.41% | 14.53% | 19.93% |
| VT – Reduced PyXtal on Reduced RRUFF | `e0sq2tus` | 7.98% | 15.13% | 19.50% |
| VT – Full PyXtal on Full RRUFF | `ollqf5gm` | 9.02% | 14.92% | 23.26% |

### Figure S3 — VT Confusion Matrix Accuracy

Accuracy on 2,795 test samples (10M_ref_ext_highres test split); checkpoint `smqmqi14` (crashed run):

| Top-1 | Top-3 | Top-5 |
|-------|-------|-------|
| 7.08% | 14.88% | 19.68% |

Note: Low accuracy is expected — the run crashed before convergence. The checkpoint illustrates confusion matrix format only (Fig. S3), not performance.

---

## RT Accuracy Reference

*(Pending — RT checkpoint mapping not yet complete.)*

---

## CNN Accuracy Reference

*(Pending — CNN checkpoint mapping not yet complete.)*
