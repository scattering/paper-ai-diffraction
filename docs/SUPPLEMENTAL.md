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
  - expected inputs: a supplemental checkpoint from `external/checkpoints/` and the corresponding RRUFF HDF5 on Stampede3 (see `reproducibility/dataset_manifest.csv`)

- [notebooks/provenance/wandb_confusion_matrix_from_table.ipynb](../notebooks/provenance/wandb_confusion_matrix_from_table.ipynb)
  - purpose: reconstruct the supplemental ViT confusion matrix (Fig S3) from the bundled W&B table artifact
  - expected input: `assets/figure_data/conf_mat_top5_copper-sweep-1_table_103_ff53214644fd32c50e63.table.json`

- [notebooks/provenance/csv_to_db_with_bravais_vit_model.ipynb](../notebooks/provenance/csv_to_db_with_bravais_vit_model.ipynb)
  - purpose: generate Fig S5 — VIT attention overlay with HKL annotations using checkpoint `pi7r8pah`
  - expected inputs: `assets/figure_data/1k_structures.csv`, `assets/figure_data/interp_metadata_clean.csv`, `external/checkpoints/xrd_model_pi7r8pah.pth`

### RT

- [notebooks/provenance/supplemental_rt_inference.ipynb](../notebooks/provenance/supplemental_rt_inference.ipynb)
  - purpose: batch inference for an RT checkpoint against a matching RRUFF benchmark HDF5; reproduces top-1/3/5 numbers for the RT rows of Tables S5/S6 (`tab:real_ablation`)
  - source: ports `flash_attn_version/test_inference.py`
  - expected inputs: an RT checkpoint at `external/checkpoints/xrd_model_<run_id>.pth` and a RRUFF intensity HDF5 (e.g. `RRUFF_low_bkg_full_intensity_cleaned.hdf5`)
  - includes optional W&B-API cell to fetch architecture from the original training run

- [notebooks/provenance/supplemental_rt_cross_test.ipynb](../notebooks/provenance/supplemental_rt_cross_test.ipynb)
  - purpose: 4×4 RT cross-evaluation matrix (Balanced / ICSD / RRUFF / Augmented training distributions × matching synthetic test sets) backing `tab:rt_distribution`
  - source: ports `flash_attn_version/cross_test.py`
  - expected inputs: the four RT checkpoints (`yv1m76u6`, `4hv17ttu`, `hwixtnv7`, `mq1l94p7`) and the four 10M synthetic HDF5 datasets on Stampede3 scratch

- [notebooks/provenance/supplemental_rt_attention.ipynb](../notebooks/provenance/supplemental_rt_attention.ipynb)
  - purpose: RT attention overlay with HKL annotations (`fig:rt_attention`); loads the eager-attention RT path so attention weights are materialised
  - source: ports `ai-diffraction/Code/Interpretability/interpret_reg_transformer.py`
  - expected inputs: checkpoint `xrd_model_hwixtnv7.pth` and pre-generated `gen_output/sample_{16,26}.{npz,json}` from `ai-diffraction/Code/Interpretability/gen_spectrum.py` (cctbx env, run separately)

### CNN

- [reproducibility/ResNet_Reproduction/reproduce_paper_results.ipynb](../reproducibility/ResNet_Reproduction/reproduce_paper_results.ipynb)
  - purpose: reproduce the CNN supplemental tables and scaling figure used in the PRX supplement
  - covered artifacts:
    - Table I
    - Fig. S1
    - Table S2
    - Table S8
    - Table S9
    - Fig. S2
  - expected inputs: CNN checkpoints from `external/checkpoints/` and the matching HDF5 files listed in `reproducibility/dataset_manifest.csv`

- [src/paper_ai_diffraction/interp-cnn/Interp_CNN_restructured.ipynb](../src/paper_ai_diffraction/interp-cnn/Interp_CNN_restructured.ipynb)
  - purpose: generate the CNN saliency figure
  - expected inputs: `src/paper_ai_diffraction/interp-cnn/xrd_model_kiifq20z.pth`, `src/paper_ai_diffraction/interp-cnn/1k_structures.csv`, `src/paper_ai_diffraction/interp-cnn/interp_metadata_clean.csv`, `src/paper_ai_diffraction/interp-cnn/SG_To_Ext_Map.pkl`
  - note: the notebook uses a manually selected example index

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

W&B reference numbers for the supplemental RT checkpoints. These are raw training/inference run values, not reproduced from bundled artifacts. W&B project: `nist-berkeley-ai-diffraction/ai-diffraction`. Checkpoint mapping is in `notebooks/provenance/supplemental_rt_inference.ipynb` (`CHECKPOINT_CONFIGS`) and `supplemental_rt_cross_test.ipynb` (`MODELS`).

### tab:rt_distribution — RT Training-Distribution Cross-Test

| Model       | W&B Run ID | depth | num_heads | spec_length | Notes |
|-------------|------------|-------|-----------|-------------|-------|
| RT-Balanced | `yv1m76u6` | 8     | 4         | 8501        | RoPE; trained on 10M PyXtal balanced (5–90°) |
| RT-ICSD     | `4hv17ttu` | 6     | 4         | 8501        | RoPE; trained on 10M PyXtal ICSD-distrib    |
| RT-RRUFF    | `hwixtnv7` | 6     | 4         | 8501        | RoPE; trained on 10M PyXtal RRUFF-distrib   |
| RT-Augmented| `mq1l94p7` | 6     | 4         | 8501        | RoPE; trained on 10M aug RRUFF-type         |

Top-1/3/5 entries of the 4×4 cross-test matrix are produced by `supplemental_rt_cross_test.ipynb`.

### tab:real_ablation — RT Real-RRUFF Ablation

| Model               | W&B Run ID | depth | num_heads | RRUFF benchmark HDF5 |
|---------------------|------------|-------|-----------|----------------------|
| RT-RRUFF (depth=6)  | `hwixtnv7` | 6     | 4         | `RRUFF_low_bkg_full_intensity_cleaned.hdf5` |
| RT 16-heads         | `7brb1pir` | 6     | 16        | `RRUFF_low_bkg_full_intensity_cleaned.hdf5` |

Additional RT rows for `tab:real_ablation` will be filled in once checkpoint mapping is finalised; the notebook’s `CHECKPOINT_CONFIGS` is the source of truth.

### fig:rt_attention — RT Attention Overlay

Checkpoint `hwixtnv7` (RT-RRUFF). Sample indices 16 and 26 from `intrep_crystal_xrd.db`; HKL/peak data pre-generated via `gen_spectrum.py` (cctbx env). Attention is read from the final transformer block via the eager `MultiHeadAttentionWithRoPE` path (the Flash Attention path never materialises the matrix).

---

## CNN Accuracy Reference

The current supplemental CNN provenance is split across the reproduction notebook, the interpretability notebook, and the archival artifact map in the Zenodo package.

Notes:
- Table S8 now explicitly uses `---` for the CNN `RRUFF` column because those runs were not completed.
