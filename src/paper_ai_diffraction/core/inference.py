import torch
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from tqdm import tqdm

from paper_ai_diffraction.core.model import VIT_model
from paper_ai_diffraction.core.dataset import get_dataloaders, get_dataloaders_test

import h5py

from collections.abc import Iterable

import sqlite3

from pymatgen.core import Lattice

# -----------------------------
# HKL annotation controls
# -----------------------------
HKL_LABEL_WINDOW_DEG = 0.5     # group HKLs by 2θ window
MAX_LABELS_PER_WINDOW = 2      # max labels per window
HKL_LABEL_Y_STEP = 40          # vertical spacing
HKL_LABEL_Y_MAX = 300          # clamp label height

from scipy.signal import find_peaks
from paper_ai_diffraction.utils.extinction_multilabel import (
    build_template_bank,
    decode_multilabel_logits,
    topk_decoded_ext_groups,
)

def parse_args():
    parser = argparse.ArgumentParser(description="ViT Inference")

    # File paths
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset (HDF5)")
    parser.add_argument("--data_path_interp", type=str, required=True, help="Path to interpretability dataset (HDF5)")
    parser.add_argument("--run_id", type=str, required=True, help="Run identifier")
    parser.add_argument("--project", type=str, required=True, help="Project name")

    # Spectra settings
    parser.add_argument("--two_theta_min", type=float, default=10.0)
    parser.add_argument("--two_theta_max", type=float, default=90.0)

    # Model hyperparameters
    parser.add_argument("--spec_length", type=int, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--num_labels", type=int, default=37)
    parser.add_argument("--patch_size", type=int, required=True)
    parser.add_argument("--embed_dim", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--drop_ratio", type=float, required=True)
    parser.add_argument("--mlp_ratio", type=float, required=True)
    parser.add_argument("--mlp_hidden_dim", type=int, required=True)
    parser.add_argument("--start_col", type=int, required=True)
    parser.add_argument("--end_col", type=int, required=True)

    # Confusion Matrix
    parser.add_argument(
        "--use_rope",
        action="store_true",
        help="Use RoPE embedding"
    )

    parser.add_argument(
        "--use_mlp",
        action="store_true",
        help="Use MLP classification head"
    )
    parser.add_argument("--label_mode", type=str, default="categorical", choices=["categorical", "multilabel"])
    parser.add_argument("--canonical_table_path", type=str, default=None)
    parser.add_argument("--final_table_path", type=str, default=None)
    parser.add_argument("--sg_lookup_path", type=str, default=None)


    # Check NaN
    parser.add_argument(
        "--check_nan",
        action="store_true",
        help="Enable NaN/Inf checks"
    )

    # Confusion Matrix
    parser.add_argument(
        "--conf_matrix",
        action="store_true",
        help="Generate confusion matrix"
    )

        # 🔴 NEW: HKL / lattice overlay inputs
    parser.add_argument(
        "--hkl_db_path",
        type=str,
        default=None,
        help="Path to SQLite database with lattice parameters"
    )

    parser.add_argument(
        "--hkl_hdf5_path",
        type=str,
        default=None,
        help="HDF5 file containing structure_id dataset"
    )

    parser.add_argument(
        "--hkl_sample_index",
        type=int,
        nargs="+",
        required=True,
        help="One or more HDF5 sample indices (0-based)"
    )

    parser.add_argument(
        "--max_hkl_index",
        type=int,
        default=3,
        help="Maximum |h|,|k|,|l| for HKL generation"
    )

    parser.add_argument(
        "--annotate_hkl",
        action="store_true",
        help="Annotate HKL indices on attention overlay plots"
    )

    

    return parser.parse_args()

def project_attention_to_2theta(attn_map, spec_length, two_theta_range):
    """
    Project attention weights back to 2θ domain.
    Ensures output length exactly matches spectrum length.
    Args:
        attn_map: torch.Tensor (num_heads, num_tokens, num_tokens)
        spec_length: int, length of spectrum (e.g. 3041)
        two_theta_range: tuple (min_2θ, max_2θ)
    Returns:
        two_theta: np.array of shape (spec_length,)
        attn_profile: np.array of shape (spec_length,)
    """
    # average attention across heads and CLS → patch tokens
    attn_cls = attn_map.mean(0)[0, 1:]  # (num_patches,)
    attn_cls = attn_cls / attn_cls.sum()

    # interpolate attention to exactly match spectrum length
    attn_profile = np.interp(
        np.arange(spec_length),
        np.linspace(0, spec_length - 1, len(attn_cls)),
        attn_cls.cpu().numpy()
    )

    # create two_theta to exactly match spec_length
    two_theta = np.linspace(two_theta_range[0], two_theta_range[1], spec_length)

    return two_theta, attn_profile


def plot_attention_overlay(xrd, attn_map, two_theta, sample_id, save_dir="./", num_bins=50, y_max=1000, hkls=None, top_n_peaks=5, annotate_offset=0.05, annotate_hkl=False):
    """
    Overlay attention map as red-gradient bins along full Y-axis.
    - xrd: 1D numpy array of intensity
    - attn_map: 1D numpy array aligned with two_theta
    - two_theta: 1D numpy array of 2θ values
    - sample_id: str/int label for saving
    - save_dir: directory to save figure
    - num_bins: number of bins for color gradient
    - y_max: maximum Y value for overlay (full spectrum range)
    """

    # -------------------------------
    # Ensure CPU numpy arrays
    # -------------------------------
    if torch.is_tensor(two_theta):
        two_theta = two_theta.detach().cpu().numpy()

    if torch.is_tensor(xrd):
        xrd = xrd.detach().cpu().numpy()

    if torch.is_tensor(attn_map):
        attn_map = attn_map.detach().cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"attention_overlay_{sample_id}.png")

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot XRD spectrum
    ax.plot(two_theta, xrd, color="black", lw=1)
    ax.set_xlabel("2θ")
    ax.set_ylabel("Intensity")
    ax.set_ylim(0, y_max)

    # Normalize attention to [0, 1]
    attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Bin the spectrum for gradient overlay
    bin_edges = np.linspace(two_theta[0], two_theta[-1], num_bins + 1)
    bin_indices = np.digitize(two_theta, bin_edges) - 1  # which bin each point belongs to

    # Plot bins with red gradient along full Y-axis
    for i in range(num_bins):
        mask = bin_indices == i
        if np.any(mask):
            ax.fill_between(two_theta[mask], 0, y_max,
                            color=(1.0, 0, 0, attn_norm[mask].mean()),  # RGBA
                            step='mid')
            
    # ----------------------------------
    # Peak-driven HKL annotation (clean)
    # ----------------------------------
    if annotate_hkl and hkls is not None and len(hkls) > 0:

        # 1️⃣ Find peaks in measured spectrum
        peak_indices, _ = find_peaks(xrd)
        if len(peak_indices) > 0:

            peak_intensities = xrd[peak_indices]

            # 2️⃣ Select top-N peaks
            top_idx = np.argsort(peak_intensities)[-top_n_peaks:][::-1]
            top_peaks = peak_indices[top_idx]

            for p in top_peaks:
                tt_peak = two_theta[p]
                y_peak = xrd[p]

                # 3️⃣ Match nearest HKL
                tt_hkl, hkl = min(hkls, key=lambda x: abs(x[0] - tt_peak))

                # 4️⃣ Annotate (NO vertical line)
                ax.text(
                    tt_peak,
                    y_peak * (1 + annotate_offset),
                    f"({hkl[0]}{hkl[1]}{hkl[2]})",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                    rotation=0,
                    zorder=4,
                )



    plt.title(f"Sample {sample_id} Attention Map")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return save_path

def plot_attention_overlay_scaled(xrd, attn_map, two_theta, sample_id, save_dir="./", num_bins=50, hkls=None, top_n_peaks=5, annotate_offset=0.05, annotate_hkl=False):

    if torch.is_tensor(two_theta):
        two_theta = two_theta.detach().cpu().numpy()
    if torch.is_tensor(xrd):
        xrd = xrd.detach().cpu().numpy()
    if torch.is_tensor(attn_map):
        attn_map = attn_map.detach().cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"attention_overlay_{sample_id}.png")

    # ── Dynamic y ceiling ──────────────────────────────────────────
    peak_max = xrd.max()
    hkl_buffer = 0.20 * peak_max if annotate_hkl else 0.10 * peak_max
    y_max = peak_max + hkl_buffer
    # ──────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(two_theta, xrd, color="black", lw=1)
    ax.set_xlabel("2θ")
    ax.set_ylabel("Intensity")
    ax.set_ylim(0, y_max)           # plot ceiling matches data

    # Normalize attention
    attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Gradient bins — fill to y_max so they span the full plot
    bin_edges = np.linspace(two_theta[0], two_theta[-1], num_bins + 1)
    bin_indices = np.digitize(two_theta, bin_edges) - 1

    for i in range(num_bins):
        mask = bin_indices == i
        if np.any(mask):
            ax.fill_between(
                two_theta[mask], 0, y_max,
                color=(1.0, 0, 0, attn_norm[mask].mean()),
                step='mid'
            )

    # HKL annotation
    if annotate_hkl and hkls is not None and len(hkls) > 0:

        peak_indices, _ = find_peaks(xrd)
        if len(peak_indices) > 0:

            peak_intensities = xrd[peak_indices]
            top_idx = np.argsort(peak_intensities)[-top_n_peaks:][::-1]
            top_peaks = peak_indices[top_idx]

            for p in top_peaks:
                tt_peak = two_theta[p]
                y_peak = xrd[p]

                tt_hkl, hkl = min(hkls, key=lambda x: abs(x[0] - tt_peak))

                ax.text(
                    tt_peak,
                    y_peak + annotate_offset * peak_max,   # offset relative to data max
                    f"({hkl[0]}{hkl[1]}{hkl[2]})",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                    rotation=0,
                    zorder=4,
                )

    plt.title(f"Sample {sample_id} Attention Map")
    plt.axis('off') # Hides ticks, labels, and spine
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return save_path

def compute_top_k_accuracy(output, target, k=1):
    """Compute top-k accuracy: fraction of samples where true label is in top-k"""
    with torch.no_grad():
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct_k = correct.any(dim=1).float().sum().item()  # only 1 per sample
        return correct_k

def check_for_nan_inf(inputs, targets, batch_idx, max_reports=5):
    """
    Scan inputs and targets for NaN/Inf values and print offending samples.
    Limits output to top N (max_reports).
    """
    reports = 0
    batch_size = inputs.size(0)

    for i in range(batch_size):
        input_has_issue = not torch.isfinite(inputs[i]).all()
        target_has_issue = not torch.isfinite(targets[i]).all()

        if input_has_issue or target_has_issue:
            print(f"\n[Batch {batch_idx}, Sample {i}] Detected issue:")
            if input_has_issue:
                print("  ⚠️ Input contains NaN/Inf")
                print(f"  Full Input Sequence:\n{inputs[i].cpu().numpy()}")
            if target_has_issue:
                print("  ⚠️ Target contains NaN/Inf")
                print(f"  Full Target:\n{targets[i].cpu().numpy()}")

            reports += 1
            if reports >= max_reports:
                print("\n[Limit Reached] Stopping after top "
                      f"{max_reports} offending samples in this batch.\n")
                return


def load_spectrum_from_dataset(dataset, sample_index):
    """
    Load spectrum and label/structure ID from a dataset (e.g., test_loader.dataset)
    """
    sample = dataset[sample_index]

    # If dataset returns (input, label)
    if isinstance(sample, (list, tuple)):
        spectrum = sample[0]  # tensor
        label_or_id = sample[1]  # could be target or structure index
    else:
        spectrum = sample
        label_or_id = sample_index  # fallback

    # Ensure batch dimension for model
    spectrum = spectrum.unsqueeze(0)  # shape: [1, spec_length]
    return spectrum, label_or_id


def load_structure_id_by_index(hdf5_path, index, dataset_name="structure_id"):
    """
    Load a structure_id from an HDF5 file by index.

    Args:
        hdf5_path (str): Path to HDF5 file containing 'structure_id' dataset.
        index (int): 0-based index of the sample.
        dataset_name (str, optional): Name of the dataset. Defaults to 'structure_id'.

    Returns:
        str: Decoded structure_id
    """
    with h5py.File(hdf5_path, "r") as f:
        if dataset_name not in f:
            raise KeyError(f"Dataset '{dataset_name}' not found in {hdf5_path}")
        
        sid = f[dataset_name][index]
        if isinstance(sid, bytes):
            sid = sid.decode("utf-8")
        
        return sid

def get_hkls_for_structure(structure_id, db_path, two_theta_min, two_theta_max, max_hkl_index=3, wavelength=1.541838):
    """
    Query lattice parameters from database and compute allowed HKLs.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT lattice_a, lattice_b, lattice_c, alpha, beta, gamma
        FROM Structures
        WHERE structure_id = ?
        """,
        (structure_id,),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        print(f"[WARN] No lattice found for structure_id={structure_id}")
        return []

    lattice = Lattice.from_parameters(*row)

    hkls = []
    for h in range(-max_hkl_index, max_hkl_index + 1):
        for k in range(-max_hkl_index, max_hkl_index + 1):
            for l in range(-max_hkl_index, max_hkl_index + 1):
                if h == k == l == 0:
                    continue
                try:
                    d = lattice.d_hkl((h, k, l))
                    sin_theta = wavelength / (2.0 * d)
                    if 0 < sin_theta <= 1:
                        tt = np.degrees(2.0 * np.arcsin(sin_theta))
                        if two_theta_min <= tt <= two_theta_max:
                            hkls.append((tt, (h, k, l)))
                except Exception:
                    continue

    hkls = sorted(hkls, key=lambda x: x[0])

    # Truncate only if user explicitly asks for a limit
    # if max_hkl_index is not None and max_hkl_index > 0:
    #     hkls = hkls[:max_hkl_index]
    
    return hkls


def main(args):
    # --- Init W&B run for inference ---
    # new run id will be provisioned, will ignore passed in run id
    run = wandb.init(
        project=args.project,  # same project name as training
        name=f"inference_{args.checkpoint.split('/')[-1].replace('.pth', '')}",
        job_type="inference",
        config={
            "checkpoint": args.checkpoint,
            "data_path": args.data_path,
            "data_path_interp": args.data_path_interp,
            "spec_length": args.spec_length,
            "patch_size": args.patch_size,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "depth": args.depth,
            "use_rope": args.use_rope,
            "use_mlp_head": args.use_mlp,
            "mlp_head_hidden_dim": args.mlp_hidden_dim
        }
    )

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load model ---
    model = VIT_model(
        spec_length=args.spec_length,
        num_output=args.num_labels if args.label_mode == "multilabel" else args.num_classes,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_ratio=args.drop_ratio,
        use_rope=args.use_rope,
        use_mlp_head=args.use_mlp,
        mlp_head_hidden_dim=args.mlp_hidden_dim
    )
    # --- Load checkpoint ---
    checkpoint = torch.load(args.checkpoint, map_location=device)

    state_dict = checkpoint.get("model", checkpoint)  # handle case where it's nested

    # Remove "module." prefix if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove "module." if it exists
        new_state_dict[name] = v

    # Load into model
    model.load_state_dict(new_state_dict, strict=False)

    # Move to device
    model = model.to(device)

    model.eval()

    # --- Load data ---
    train_loader, val_loader, test_loader, num_classes, intensity_points = get_dataloaders_test(
        args.data_path,
        batch_size=128,
        num_classes=args.num_classes,
        prefetch_factor=4,
        start_col=args.start_col,
        end_col=args.end_col,
        label_mode=args.label_mode,
        canonical_table_path=args.canonical_table_path,
        final_table_path=args.final_table_path,
        sg_lookup_path=args.sg_lookup_path,
    )

    print(f"[DEBUG] num_classes={num_classes}, intensity_points={intensity_points}")
    
    if(args.check_nan):

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 🔍 Run NaN/Inf check (limit to 5 samples per batch)
            check_for_nan_inf(inputs, targets, batch_idx, max_reports=5)

    dataset = test_loader.dataset

    for idx in args.hkl_sample_index:
        # 1️⃣ Load spectrum & label/structure ID from dataset
        spectrum, structure_idx = load_spectrum_from_dataset(dataset, idx)
        spectrum = spectrum.to(device)

        # 2️⃣ Run model to get attention
        with torch.no_grad():
            logits, attn_maps = model.forward_with_attn(spectrum)

        last_attn = attn_maps[-1]
        if isinstance(last_attn, (list, tuple)):
            last_attn = last_attn[0]
        if last_attn.ndim == 4:  # B, heads, N, N
            last_attn = last_attn[0]

        # 3️⃣ Project attention to spectrum length
        two_theta, attn_profile = project_attention_to_2theta(
            last_attn,
            spec_length=spectrum.shape[1],
            two_theta_range=(args.two_theta_min, args.two_theta_max)
        )

        # # 4️⃣ Lookup structure_id from interp HDF5
        # with h5py.File(args.hkl_hdf5_path, "r") as f:
        #     structure_id = f["structure_id"][structure_idx]
        #     if isinstance(structure_id, bytes):
        #         structure_id = structure_id.decode("utf-8")

        structure_id = load_structure_id_by_index(args.data_path_interp, structure_idx, "structure_id")

        # 5️⃣ Compute HKLs for this structure
        hkls = get_hkls_for_structure(
            structure_id,
            args.hkl_db_path,
            args.two_theta_min,
            args.two_theta_max,
            args.max_hkl_index
        )

        # 6️⃣ Plot overlay
        save_path = plot_attention_overlay(
            xrd=spectrum.squeeze().cpu().numpy(),
            attn_map=attn_profile,
            two_theta=two_theta,
            sample_id=idx,
            hkls=hkls,
            save_dir="./attention_plots",
            annotate_hkl=args.annotate_hkl,  # controlled by flag
        )


        attn_profile_artifact = wandb.Artifact(f"attention_profiles_{run.name}", type="attention")

        for idx in args.hkl_sample_index:
            # ... existing code ...
            
            np.save(f"/tmp/attn_profile_{idx}.npy", attn_profile)
            attn_profile_artifact.add_file(f"/tmp/attn_profile_{idx}.npy", name=f"attn_profile_{idx}.npy")

        wandb.log_artifact(attn_profile_artifact)

        wandb.log({f"attention_overlay_sample_{idx}": wandb.Image(save_path)})

    if(args.conf_matrix):
        criterion = nn.BCEWithLogitsLoss() if args.label_mode == "multilabel" else nn.CrossEntropyLoss()
        template_bank = None
        ext_group_order = None
        if args.label_mode == "multilabel":
            template_bank, ext_group_order, _ = build_template_bank(
                canonical_table_path=args.canonical_table_path,
                final_table_path=args.final_table_path,
                sg_lookup_path=args.sg_lookup_path,
            )
        test_loss, correct, total = 0, 0, 0
        top1_correct, top3_correct, top5_correct = 0, 0, 0

        # Confusion Matrix
        all_preds = []
        all_targets = []
        all_top5_preds = []

        # # Embedding Visualization
        # cls_embeddings = []
        # cls_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
                if args.label_mode == "multilabel":
                    inputs, targets, ext_targets = batch
                    ext_targets = ext_targets.to(device)
                else:
                    inputs, targets = batch
                    ext_targets = targets
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                if args.label_mode == "multilabel":
                    predicted = decode_multilabel_logits(outputs, template_bank, ext_group_order) - 1
                else:
                    _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == ext_targets).sum().item()

                if args.label_mode == "multilabel":
                    target_ext = ext_targets + 1
                    top1_correct += (topk_decoded_ext_groups(outputs, template_bank, ext_group_order, 1).squeeze(1) == target_ext).sum().item()
                    top3_correct += (topk_decoded_ext_groups(outputs, template_bank, ext_group_order, 3) == target_ext.unsqueeze(1)).any(dim=1).sum().item()
                    top5_correct += (topk_decoded_ext_groups(outputs, template_bank, ext_group_order, 5) == target_ext.unsqueeze(1)).any(dim=1).sum().item()
                else:
                    top1_correct += compute_top_k_accuracy(outputs, targets, k=1)
                    top3_correct += compute_top_k_accuracy(outputs, targets, k=3)
                    top5_correct += compute_top_k_accuracy(outputs, targets, k=5)

                # Collect predictions and targets for confusion matrix
                all_preds.append(predicted.cpu().numpy())
                all_targets.append(ext_targets.cpu().numpy())

                # For top-5 confusion matrix:
                if args.label_mode == "multilabel":
                    top5_preds = topk_decoded_ext_groups(outputs, template_bank, ext_group_order, 5).cpu().numpy() - 1
                    targets_np = ext_targets.cpu().numpy()
                else:
                    top5_preds = torch.topk(outputs, k=5, dim=1).indices.cpu().numpy()
                    targets_np = targets.cpu().numpy()
                top5_selected_preds = []
                for i in range(targets_np.shape[0]):
                    if targets_np[i] in top5_preds[i]:
                        # pick ground truth label as predicted (simulate "correct")
                        top5_selected_preds.append(targets_np[i])
                    else:
                        # pick top-1 prediction (simulate "incorrect")
                        top5_selected_preds.append(top5_preds[i][0])
                all_top5_preds.append(np.array(top5_selected_preds))

                # # Embedding visualization
                # cls_emb = model(inputs, return_cls_embedding=True)
                # cls_embeddings.append(cls_emb.cpu())
                # cls_labels.append(targets.cpu())


        # Accuracy (make sure to divide by total!)
        test_accuracy = 100 * correct / total
        top1_accuracy = 100 * top1_correct / total
        top3_accuracy = 100 * top3_correct / total
        top5_accuracy = 100 * top5_correct / total

        print(f"Total samples: {total}")

        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
        print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")
        print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

        # Concatenate all predictions and targets for confusion matrix
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_top5_preds = np.concatenate(all_top5_preds)

        # Log final test metrics
        wandb.log({
            "total_samples": total,
            "test_loss": test_loss,
            "test_accuracy_top1": top1_accuracy,
            "test_accuracy_top3": top3_accuracy,
            "test_accuracy_top5": top5_accuracy,
        })

    #        print(all_targets.shape)
    #        print(all_top5_preds.shape)

        # Log confusion matrix to W&B
        wandb.log({
            f"conf_mat_top1_{wandb.run.name}": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_targets,
                preds=all_preds,
                class_names=[str(i) for i in range(num_classes)]
            )
            # f"conf_mat_top5_{wandb.run.name}": wandb.plot.confusion_matrix(
            #     probs=None,
            #     y_true=all_targets,
            #     preds=all_top5_preds,
            #     class_names=[str(i) for i in range(config["num_classes"])]
            # )
        })
            
        print("✅ Logged confusion matrix to W&B")


if __name__ == "__main__":
    args = parse_args()
    main(args)
