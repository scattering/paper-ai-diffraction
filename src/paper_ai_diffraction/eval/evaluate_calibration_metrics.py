#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from paper_ai_diffraction.core.dataset import get_dataloaders_test
from paper_ai_diffraction.utils.extinction_multilabel import (
    build_template_bank,
    template_mask_from_ext_groups,
)
from paper_ai_diffraction.core.model import VIT_model, adapt_patch_embed_input_channels


def parse_args():
    parser = argparse.ArgumentParser(description="Compute calibration metrics and bootstrap CIs for aux-head decoding.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--prior-data-path", required=True)
    parser.add_argument("--aux-temperature", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_config(args, checkpoint):
    config = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    if args.config:
        with open(args.config, "r") as handle:
            config.update(json.load(handle))
    config["data_path"] = args.eval_data_path
    return config


def build_model(config, checkpoint, device):
    model = VIT_model(
        spec_length=config["spec_length"],
        num_output=config.get("num_labels", 37),
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        drop_ratio=config["dropout"],
        use_rope=config["use_rope"],
        use_mlp_head=config["use_mlp_head"],
        mlp_head_hidden_dim=config["mlp_head_hidden_dim"],
        use_physics_pe=config.get("use_physics_pe", False),
        physics_pe_mode=config.get("physics_pe_mode", "sin2theta"),
        two_theta_min=float(config.get("two_theta_min", 5.0)),
        two_theta_max=float(config.get("two_theta_max", 90.0)),
        physics_pe_scale=float(config.get("physics_pe_scale", 1.0)),
        use_coordinate_channel=config.get("use_coordinate_channel", False),
        coordinate_mode=config.get("coordinate_mode", "sin2theta"),
    )
    if config.get("use_split_head", False):
        model.head_sys = nn.Linear(model.embed_dim, 7)
        model.head_lat = nn.Linear(model.embed_dim, 5)
        model.head_ops = nn.Linear(model.embed_dim, 25)
    if config.get("use_aux_ext_head", False):
        model.aux_ext_head = nn.Linear(model.embed_dim, config.get("num_classes", 99))
        aux_state = checkpoint.get("aux_ext_head_state_dict") if isinstance(checkpoint, dict) else None
        if aux_state is not None:
            model.aux_ext_head.load_state_dict(aux_state)
    adapted_state_dict = adapt_patch_embed_input_channels(get_model_state_dict(checkpoint), model)
    model.load_state_dict(adapted_state_dict, strict=False)
    return model.to(device).eval()


def load_seen_ext_groups(h5_path):
    import h5py

    with h5py.File(h5_path, "r") as handle:
        return sorted(set(int(v) for v in handle["y_train"][:].tolist()))


def load_log_priors(h5_path, ext_group_order, smoothing=1.0):
    import h5py

    with h5py.File(h5_path, "r") as handle:
        y_train = handle["y_train"][:]
    counts = {int(ext): smoothing for ext in ext_group_order}
    for value in y_train.tolist():
        counts[int(value)] = counts.get(int(value), smoothing) + 1.0
    probs = torch.tensor([counts[int(ext)] for ext in ext_group_order], dtype=torch.float32)
    probs = probs / probs.sum()
    return torch.log(probs)


def multiclass_brier(probs: np.ndarray, targets: np.ndarray) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(targets.shape[0]), targets] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (conf >= left) & (conf <= right)
        else:
            mask = (conf >= left) & (conf < right)
        if not np.any(mask):
            continue
        acc = correct[mask].mean()
        c = conf[mask].mean()
        ece += mask.mean() * abs(acc - c)
    return float(ece)


def summarize_probs(probs: np.ndarray, targets: np.ndarray) -> dict:
    top1 = float(np.mean(probs.argmax(axis=1) == targets))
    top5_idx = np.argpartition(-probs, kth=4, axis=1)[:, :5]
    top5 = float(np.mean([t in row for t, row in zip(targets.tolist(), top5_idx.tolist())]))
    true_probs = probs[np.arange(targets.shape[0]), targets]
    nll = float(np.mean(-np.log(np.clip(true_probs, 1e-12, 1.0))))
    return {
        "top1": top1,
        "top5": top5,
        "ece": expected_calibration_error(probs, targets),
        "nll": nll,
        "brier": multiclass_brier(probs, targets),
    }


def bootstrap_summary(probs: np.ndarray, targets: np.ndarray, rounds: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = targets.shape[0]
    stats = {"top1": [], "top5": [], "ece": [], "nll": [], "brier": []}
    for _ in range(rounds):
        idx = rng.integers(0, n, size=n)
        sub = summarize_probs(probs[idx], targets[idx])
        for key in stats:
            stats[key].append(sub[key])
    out = {}
    for key, vals in stats.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
        }
    return out


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_config(args, checkpoint)
    device = torch.device(args.device)

    model = build_model(config, checkpoint, device)
    _, ext_group_order, _ = build_template_bank(
        canonical_table_path=config.get("canonical_table_path"),
        final_table_path=config.get("final_table_path"),
        sg_lookup_path=config.get("sg_lookup_path"),
    )
    log_priors = load_log_priors(args.prior_data_path, ext_group_order).to(device)
    seen_mask = template_mask_from_ext_groups(ext_group_order, load_seen_ext_groups(args.prior_data_path)).to(device)

    _, _, test_loader, _, _ = get_dataloaders_test(
        h5_file=args.eval_data_path,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        num_classes=config.get("num_classes", 99),
        prefetch_factor=config.get("prefetch_factor", 4) if args.num_workers > 0 else None,
        start_col=config["start_col"],
        end_col=config["end_col"],
        label_mode=config.get("label_mode", "multilabel"),
        canonical_table_path=config.get("canonical_table_path"),
        final_table_path=config.get("final_table_path"),
        sg_lookup_path=config.get("sg_lookup_path"),
        max_samples_test=config.get("max_samples_test"),
    )

    raw_probs = []
    bayes_probs = []
    all_targets = []
    with torch.no_grad():
        for inputs, _, ext_targets in test_loader:
            inputs = inputs.to(device)
            ext_targets = ext_targets.to(device)
            cls_embedding = model(inputs, return_cls_embedding=True)
            aux_logits = model.aux_ext_head(cls_embedding)
            raw = torch.softmax(aux_logits, dim=1)
            bayes_scores = aux_logits / args.aux_temperature + log_priors.to(dtype=aux_logits.dtype).unsqueeze(0)
            bayes_scores = bayes_scores.masked_fill(~seen_mask.unsqueeze(0), float("-inf"))
            bayes = torch.softmax(bayes_scores, dim=1)
            raw_probs.append(raw.cpu().numpy())
            bayes_probs.append(bayes.cpu().numpy())
            all_targets.append(ext_targets.cpu().numpy())

    raw_probs = np.concatenate(raw_probs, axis=0)
    bayes_probs = np.concatenate(bayes_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    output = {
        "checkpoint": args.checkpoint,
        "eval_data_path": args.eval_data_path,
        "prior_data_path": args.prior_data_path,
        "aux_temperature": args.aux_temperature,
        "n_samples": int(targets.shape[0]),
        "uncalibrated_aux": {
            "point_estimate": summarize_probs(raw_probs, targets),
            "bootstrap": bootstrap_summary(raw_probs, targets, args.bootstrap, args.seed),
        },
        "calibrated_bayesian_aux": {
            "point_estimate": summarize_probs(bayes_probs, targets),
            "bootstrap": bootstrap_summary(bayes_probs, targets, args.bootstrap, args.seed + 1),
        },
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
