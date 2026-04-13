#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders_test
from extinction_multilabel import build_template_bank
from model import VIT_model, adapt_patch_embed_input_channels


def parse_args():
    parser = argparse.ArgumentParser(description="Measure exact split-head bit-pattern validity rates.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


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
    model.head_sys = nn.Linear(model.embed_dim, 7)
    model.head_lat = nn.Linear(model.embed_dim, 5)
    model.head_ops = nn.Linear(model.embed_dim, 25)
    adapted = adapt_patch_embed_input_channels(get_model_state_dict(checkpoint), model)
    model.load_state_dict(adapted, strict=False)
    return model.to(device).eval()


def main():
    args = parse_args()
    with open(args.config) as handle:
        config = json.load(handle)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    merged = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    merged.update(config)
    merged["data_path"] = args.eval_data_path

    device = torch.device(args.device)
    model = build_model(merged, checkpoint, device)
    template_bank, ext_group_order, _ = build_template_bank(
        canonical_table_path=merged.get("canonical_table_path"),
        final_table_path=merged.get("final_table_path"),
        sg_lookup_path=merged.get("sg_lookup_path"),
    )
    template_bank = template_bank.to(device=device, dtype=torch.float32)

    _, _, test_loader, _, _ = get_dataloaders_test(
        h5_file=args.eval_data_path,
        batch_size=merged["batch_size"],
        num_workers=args.num_workers,
        num_classes=merged.get("num_classes", 99),
        prefetch_factor=merged.get("prefetch_factor", 4) if args.num_workers > 0 else None,
        start_col=merged["start_col"],
        end_col=merged["end_col"],
        label_mode=merged.get("label_mode", "multilabel"),
        canonical_table_path=merged.get("canonical_table_path"),
        final_table_path=merged.get("final_table_path"),
        sg_lookup_path=merged.get("sg_lookup_path"),
        max_samples_test=merged.get("max_samples_test"),
    )

    counts = {
        "n": 0,
        "valid_single_match": 0,
        "ambiguous_multi_match": 0,
        "invalid_zero_match": 0,
        "single_match_correct": 0,
    }
    examples = {"ambiguous": [], "invalid": []}

    with torch.no_grad():
        sample_offset = 0
        for inputs, _, ext_targets in test_loader:
            inputs = inputs.to(device)
            ext_targets = ext_targets.to(device)
            cls = model(inputs, return_cls_embedding=True)
            sys_logits = model.head_sys(cls)
            lat_logits = model.head_lat(cls)
            op_logits = model.head_ops(cls)

            sys_onehot = torch.zeros_like(sys_logits).scatter_(1, torch.argmax(sys_logits, dim=1, keepdim=True), 1.0)
            lat_onehot = torch.zeros_like(lat_logits).scatter_(1, torch.argmax(lat_logits, dim=1, keepdim=True), 1.0)
            op_bits = (torch.sigmoid(op_logits) >= 0.5).to(dtype=torch.float32)
            bitvec = torch.cat([sys_onehot, lat_onehot, op_bits], dim=1)
            matches = (bitvec.unsqueeze(1) == template_bank.unsqueeze(0)).all(dim=2)
            match_counts = matches.sum(dim=1)

            counts["n"] += int(ext_targets.shape[0])
            counts["valid_single_match"] += int((match_counts == 1).sum().item())
            counts["ambiguous_multi_match"] += int((match_counts > 1).sum().item())
            counts["invalid_zero_match"] += int((match_counts == 0).sum().item())

            single_rows = torch.nonzero(match_counts == 1, as_tuple=False).flatten()
            if single_rows.numel() > 0:
                matched_idx = torch.argmax(matches[single_rows].to(torch.int64), dim=1)
                pred_ext = torch.tensor(ext_group_order, device=device, dtype=torch.long)[matched_idx] - 1
                counts["single_match_correct"] += int((pred_ext == ext_targets[single_rows]).sum().item())

            for row in torch.nonzero(match_counts > 1, as_tuple=False).flatten()[:5]:
                examples["ambiguous"].append(
                    {
                        "index": sample_offset + int(row.item()),
                        "true_eg": int(ext_targets[row].item()) + 1,
                        "matched_egs": [int(ext_group_order[i]) for i in torch.nonzero(matches[row], as_tuple=False).flatten().tolist()],
                    }
                )
            for row in torch.nonzero(match_counts == 0, as_tuple=False).flatten()[:5]:
                examples["invalid"].append(
                    {
                        "index": sample_offset + int(row.item()),
                        "true_eg": int(ext_targets[row].item()) + 1,
                    }
                )
            sample_offset += ext_targets.shape[0]

    n = max(counts["n"], 1)
    output = {
        "checkpoint": args.checkpoint,
        "eval_data_path": args.eval_data_path,
        "counts": counts,
        "rates_pct": {
            "valid_single_match": 100.0 * counts["valid_single_match"] / n,
            "ambiguous_multi_match": 100.0 * counts["ambiguous_multi_match"] / n,
            "invalid_zero_match": 100.0 * counts["invalid_zero_match"] / n,
            "single_match_correct_within_all": 100.0 * counts["single_match_correct"] / n,
        },
        "examples": examples,
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
