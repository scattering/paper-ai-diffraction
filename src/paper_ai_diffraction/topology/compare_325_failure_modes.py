#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict

import h5py
import torch
import torch.nn as nn
from tqdm import tqdm

from paper_ai_diffraction.core.dataset import get_dataloaders_test
from paper_ai_diffraction.utils.extinction_multilabel import (
    build_template_bank,
    score_split_head_templates,
    template_mask_from_ext_groups,
)
from paper_ai_diffraction.core.model import VIT_model, adapt_patch_embed_input_channels


def parse_args():
    parser = argparse.ArgumentParser(description="Compare failure modes on the 325 benchmark across checkpoints.")
    parser.add_argument("--specs-json", required=True, help="JSON file listing model specs.")
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--prior-data-path", required=True)
    parser.add_argument("--output-json", required=True)
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


def load_config(config_path, checkpoint, eval_data_path):
    config = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    with open(config_path, "r") as handle:
        config.update(json.load(handle))
    config["data_path"] = eval_data_path
    return config


def load_log_priors(h5_path, ext_group_order, smoothing=1.0):
    with h5py.File(h5_path, "r") as handle:
        y_train = handle["y_train"][:]
    counts = {int(ext): smoothing for ext in ext_group_order}
    for value in y_train.tolist():
        counts[int(value)] = counts.get(int(value), smoothing) + 1.0
    probs = torch.tensor([counts[int(ext)] for ext in ext_group_order], dtype=torch.float32)
    probs = probs / probs.sum()
    return torch.log(probs)


def load_seen_ext_groups(h5_path):
    with h5py.File(h5_path, "r") as handle:
        return sorted(set(int(v) for v in handle["y_train"][:].tolist()))


def summarize_counter(counter, topn=10):
    return [{"eg": int(k), "count": int(v)} for k, v in counter.most_common(topn)]


def main():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.specs_json, "r") as handle:
        specs = json.load(handle)

    first_checkpoint = torch.load(specs[0]["checkpoint"], map_location="cpu")
    first_config = load_config(specs[0]["config"], first_checkpoint, args.eval_data_path)
    _, _, test_loader, _, _ = get_dataloaders_test(
        h5_file=first_config["data_path"],
        batch_size=first_config["batch_size"],
        num_workers=first_config.get("num_workers", 8),
        num_classes=first_config.get("num_classes", 99),
        prefetch_factor=first_config.get("prefetch_factor", 4),
        start_col=first_config["start_col"],
        end_col=first_config["end_col"],
        label_mode=first_config.get("label_mode", "multilabel"),
        canonical_table_path=first_config.get("canonical_table_path"),
        final_table_path=first_config.get("final_table_path"),
        sg_lookup_path=first_config.get("sg_lookup_path"),
        max_samples_test=first_config.get("max_samples_test"),
    )
    template_bank, ext_group_order, _ = build_template_bank(
        canonical_table_path=first_config.get("canonical_table_path"),
        final_table_path=first_config.get("final_table_path"),
        sg_lookup_path=first_config.get("sg_lookup_path"),
    )
    template_bank = template_bank.to(device)
    seen_ext_groups = load_seen_ext_groups(args.prior_data_path)
    seen_mask = template_mask_from_ext_groups(ext_group_order, seen_ext_groups).to(device)
    log_priors = load_log_priors(args.prior_data_path, ext_group_order).to(device)
    ext_group_tensor = torch.tensor(ext_group_order, device=device, dtype=torch.long) - 1

    models = []
    for spec in specs:
        checkpoint = torch.load(spec["checkpoint"], map_location="cpu")
        config = load_config(spec["config"], checkpoint, args.eval_data_path)
        models.append(
            {
                "name": spec["name"],
                "decoder": spec.get("decoder", "aux_bayes"),
                "temperature": float(spec.get("temperature", 1.0)),
                "checkpoint": spec["checkpoint"],
                "config": spec["config"],
                "model": build_model(config, checkpoint, device),
            }
        )

    outputs = {}
    for item in models:
        outputs[item["name"]] = {
            "correct": 0,
            "total": 0,
            "wrong_pred_counts": Counter(),
            "top1_pred_counts": Counter(),
            "true_on_wrong_counts": Counter(),
            "per_example": [],
        }

    with torch.no_grad():
        sample_idx = 0
        for inputs, _, ext_targets in tqdm(test_loader, desc="Failure-mode compare"):
            inputs = inputs.to(device)
            ext_targets = ext_targets.to(device)
            batch = ext_targets.size(0)
            cls_embeddings = {}
            for item in models:
                cls_embeddings[item["name"]] = item["model"](inputs, return_cls_embedding=True)

            for item in models:
                cls = cls_embeddings[item["name"]]
                if item["decoder"] == "split_bayes":
                    logits = torch.cat(
                        [item["model"].head_sys(cls), item["model"].head_lat(cls), item["model"].head_ops(cls)],
                        dim=1,
                    )
                    scores = score_split_head_templates(
                        logits,
                        template_bank,
                        allowed_mask=seen_mask,
                        log_priors=log_priors,
                        prior_weight=1.0,
                        impossible_operator_masking=True,
                        impossible_operator_prob=1e-3,
                    )
                    scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, -1e9))
                    topk_idx = torch.topk(scores, k=5, dim=1).indices
                    topk = ext_group_tensor[topk_idx]
                    top1 = topk[:, 0]
                else:
                    aux_logits = item["model"].aux_ext_head(cls)
                    scaled = aux_logits / item["temperature"]
                    scores = scaled + log_priors.unsqueeze(0)
                    topk = torch.topk(scores, k=5, dim=1).indices
                    top1 = topk[:, 0]

                out = outputs[item["name"]]
                out["total"] += batch
                out["correct"] += (top1 == ext_targets).sum().item()
                for row in range(batch):
                    target = int(ext_targets[row].item()) + 1
                    pred = int(top1[row].item()) + 1
                    out["top1_pred_counts"][pred] += 1
                    if pred != target:
                        out["wrong_pred_counts"][pred] += 1
                        out["true_on_wrong_counts"][target] += 1
                    out["per_example"].append(
                        {
                            "index": sample_idx + row,
                            "true_eg": target,
                            "pred_eg": pred,
                            "top5_eg": [int(v) + 1 for v in topk[row].tolist()],
                            "correct": pred == target,
                        }
                    )
            sample_idx += batch

    summary = {"eval_data_path": args.eval_data_path, "prior_data_path": args.prior_data_path, "models": {}}
    for name, out in outputs.items():
        summary["models"][name] = {
            "accuracy_top1_pct": 100.0 * out["correct"] / out["total"],
            "total": out["total"],
            "top1_pred_counts": summarize_counter(out["top1_pred_counts"]),
            "wrong_pred_counts": summarize_counter(out["wrong_pred_counts"]),
            "true_on_wrong_counts": summarize_counter(out["true_on_wrong_counts"]),
            "eg98_wrong_count": int(out["wrong_pred_counts"].get(98, 0)),
            "eg14_wrong_count": int(out["wrong_pred_counts"].get(14, 0)),
            "examples": out["per_example"],
        }

    with open(args.output_json, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({k: {"accuracy_top1_pct": v["accuracy_top1_pct"], "eg98_wrong_count": v["eg98_wrong_count"], "eg14_wrong_count": v["eg14_wrong_count"]} for k, v in summary["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
