import torch
import wandb
import json
import argparse
from paper_ai_diffraction.core.model import VIT_model, adapt_patch_embed_input_channels
from paper_ai_diffraction.core.dataset import get_dataloaders, get_mixed_dataloaders
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torchinfo import summary
import io
import contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import unicodedata
import os
import time
import socket
from functools import partial
import h5py
from torchvision.ops import sigmoid_focal_loss

from paper_ai_diffraction.utils.extinction_multilabel import (
    build_template_bank,
    decode_multilabel_logits,
    decode_split_head_logits,
    ext_group_to_multilabel_target,
    multilabel_targets_to_split_targets,
    topk_decoded_ext_groups,
    topk_decoded_split_head_ext_groups,
)
from paper_ai_diffraction.core.streaming_dataset import StreamingConfig, get_streaming_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("config_path", nargs="?", help="Optional positional path to the configuration file")
    parser.add_argument("--config", default=None, help="Path to the configuration file")
    parser.add_argument("--sweep", action="store_true", help="Flag to indicate if this is a sweep run")
    parser.add_argument("--distributed", action="store_true", help="Enable DistributedDataParallel (DDP)")
    parser.add_argument("--node_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument("--sweep_id", type=str, default="", help="W&B Sweep ID")
    parser.add_argument("--num_sweeps", type=int, default=1, help="W&B Number of Sweeps")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()
    if args.config is None:
        args.config = args.config_path
    if args.config is None:
        parser.error("the following arguments are required: --config")
    return args

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return materialize_config(config)


def materialize_config(config):
    materialized = dict(config)
    parameters = config.get("parameters", {})
    for key, value in parameters.items():
        if not isinstance(value, dict):
            materialized[key] = value
            continue
        if "value" in value:
            materialized[key] = value["value"]
        elif "values" in value and value["values"]:
            materialized[key] = value["values"][0]
    return materialized

def clean_summary_str(summary_str):
    # Normalize and strip non-ASCII characters
    normalized = unicodedata.normalize("NFKD", summary_str)
    return normalized.encode("ascii", "ignore").decode("ascii")


def get_nested_config_value(config, key, default=None):
    if key in config:
        return config[key]
    return config.get("parameters", {}).get(key, {}).get("value", default)


def get_label_mode(config):
    return get_nested_config_value(config, "label_mode", "categorical")


def get_model_output_dim(config):
    if get_label_mode(config) == "multilabel":
        return get_nested_config_value(config, "num_labels", 37)
    return get_nested_config_value(config, "num_classes", 99)


def use_aux_ext_head(config):
    return bool(get_nested_config_value(config, "use_aux_ext_head", False))


def use_split_head(config):
    return bool(get_nested_config_value(config, "use_split_head", False))


def get_aux_ext_weight(config):
    return float(get_nested_config_value(config, "aux_ext_weight", 1.0))


def get_resume_checkpoint(config):
    return get_nested_config_value(config, "resume_checkpoint", None)


def get_resume_weights_only(config):
    return bool(get_nested_config_value(config, "resume_weights_only", False))


def get_checkpoint_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def forward_model_outputs(model, inputs, config):
    if get_label_mode(config) != "multilabel":
        logits = model(inputs)
        return logits, None

    cls_embedding = model(inputs, return_cls_embedding=True)
    model_ref = model.module if isinstance(model, DDP) else model
    if use_split_head(config):
        multilabel_logits = torch.cat(
            [
                model_ref.head_sys(cls_embedding),
                model_ref.head_lat(cls_embedding),
                model_ref.head_ops(cls_embedding),
            ],
            dim=1,
        )
    else:
        multilabel_logits = model_ref.head(cls_embedding)
    aux_logits = model_ref.aux_ext_head(cls_embedding) if use_aux_ext_head(config) else None
    return multilabel_logits, aux_logits


def unpack_batch(batch, label_mode):
    if label_mode == "multilabel":
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], batch[1]


def compute_primary_loss(outputs, targets, label_mode, criterion, config, split_criteria=None):
    if label_mode != "multilabel" or not use_split_head(config):
        return criterion(outputs, targets)

    system_targets, lattice_targets, operator_targets = multilabel_targets_to_split_targets(targets)
    sys_logits = outputs[:, 0:7]
    lat_logits = outputs[:, 7:12]
    op_logits = outputs[:, 12:]
    return (
        split_criteria["system"](sys_logits, system_targets)
        + split_criteria["lattice"](lat_logits, lattice_targets)
        + split_criteria["operators"](op_logits, operator_targets)
    )


def compute_multilabel_topk_accuracy(logits, ext_targets, template_bank, ext_group_order, k):
    decoded = topk_decoded_ext_groups(logits, template_bank, ext_group_order, k=k)
    target_ext = ext_targets + 1
    return (decoded == target_ext.unsqueeze(1)).any(dim=1).float().sum().item()


def compute_split_head_topk_accuracy(logits, ext_targets, template_bank, ext_group_order, k):
    decoded = topk_decoded_split_head_ext_groups(logits, template_bank, ext_group_order, k=k)
    target_ext = ext_targets + 1
    return (decoded == target_ext.unsqueeze(1)).any(dim=1).float().sum().item()


def build_operator_pos_weight(config, templates):
    if not get_nested_config_value(config, "use_operator_pos_weight", False):
        return None
    if get_nested_config_value(config, "data_mode", "hdf5") != "hdf5":
        return None

    data_path = get_nested_config_value(config, "data_path", None)
    if not data_path:
        return None

    with h5py.File(data_path, "r") as h5:
        ext_groups = torch.from_numpy(h5["y_train"][:]).long()

    targets = torch.stack(
        [ext_group_to_multilabel_target(int(ext_group), templates) for ext_group in ext_groups],
        dim=0,
    )
    _, _, operator_targets = multilabel_targets_to_split_targets(targets)
    positives = operator_targets.sum(dim=0)
    total = torch.tensor(float(operator_targets.shape[0]), dtype=torch.float32)
    negatives = total - positives
    pos_weight = negatives / positives.clamp_min(1.0)
    pos_weight_mode = str(get_nested_config_value(config, "operator_pos_weight_mode", "raw")).lower()
    if pos_weight_mode == "sqrt":
        pos_weight = torch.sqrt(pos_weight)
    elif pos_weight_mode != "raw":
        raise ValueError(f"Unsupported operator_pos_weight_mode: {pos_weight_mode}")

    max_pos_weight = get_nested_config_value(config, "operator_pos_weight_max", None)
    if max_pos_weight is not None:
        pos_weight = pos_weight.clamp_max(float(max_pos_weight))
    return pos_weight.to(dtype=torch.float32)


def build_operator_alpha(config, templates):
    if get_nested_config_value(config, "data_mode", "hdf5") != "hdf5":
        return None

    data_path = get_nested_config_value(config, "data_path", None)
    if not data_path:
        return None

    with h5py.File(data_path, "r") as h5:
        ext_groups = torch.from_numpy(h5["y_train"][:]).long()

    targets = torch.stack(
        [ext_group_to_multilabel_target(int(ext_group), templates) for ext_group in ext_groups],
        dim=0,
    )
    _, _, operator_targets = multilabel_targets_to_split_targets(targets)
    prevalence = operator_targets.float().mean(dim=0)

    alpha_mode = str(get_nested_config_value(config, "operator_focal_alpha_mode", "none")).lower()
    if alpha_mode == "none":
        return None
    if alpha_mode == "prevalence_complement":
        alpha = 1.0 - prevalence
        alpha = alpha.clamp(min=1e-4, max=1.0 - 1e-4)
        return alpha.to(dtype=torch.float32)
    raise ValueError(f"Unsupported operator_focal_alpha_mode: {alpha_mode}")


def build_operator_loss(config, templates, device, rank):
    operator_loss_type = str(get_nested_config_value(config, "operator_loss_type", "bce")).lower()

    if operator_loss_type == "bce":
        operator_pos_weight = build_operator_pos_weight(config, templates)
        if operator_pos_weight is not None and rank == 0:
            print(
                "Using operator pos_weight:",
                ", ".join(f"{v:.2f}" for v in operator_pos_weight.tolist()),
            )
        return nn.BCEWithLogitsLoss(
            pos_weight=operator_pos_weight.to(device) if operator_pos_weight is not None else None
        )

    if operator_loss_type == "focal":
        operator_alpha = build_operator_alpha(config, templates)
        gamma = float(get_nested_config_value(config, "operator_focal_gamma", 2.0))
        loss_weight = float(get_nested_config_value(config, "operator_focal_loss_weight", 1.0))

        if operator_alpha is not None and rank == 0:
            print(
                "Using operator focal alpha:",
                ", ".join(f"{v:.2f}" for v in operator_alpha.tolist()),
            )
        if rank == 0:
            print(f"Using operator focal gamma: {gamma:.2f}, loss_weight: {loss_weight:.2f}")

        operator_alpha = operator_alpha.to(device) if operator_alpha is not None else None

        def focal_operator_loss(logits, targets):
            targets = targets.float()
            per_element = sigmoid_focal_loss(
                logits,
                targets,
                alpha=-1.0,
                gamma=gamma,
                reduction="none",
            )
            if operator_alpha is not None:
                alpha_factor = torch.where(targets > 0.5, operator_alpha.unsqueeze(0), 1.0 - operator_alpha.unsqueeze(0))
                per_element = per_element * alpha_factor
            return loss_weight * per_element.mean()

        return focal_operator_loss

    raise ValueError(f"Unsupported operator_loss_type: {operator_loss_type}")


def compute_top_k_accuracy(output, target, k=1):
    """Compute top-k accuracy: fraction of samples where true label is in top-k"""
    with torch.no_grad():
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct_k = correct.any(dim=1).float().sum().item()  # only 1 per sample
        return correct_k

def wait_for_master():
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    rank = int(os.environ.get("RANK", "0"))

    if rank == 0:
        print(f"[rank{rank}] Master node, skipping connectivity test.")
        return

    for attempt in range(10):
        try:
            with socket.create_connection((master_addr, int(master_port)), timeout=3):
                print(f"[rank{rank}] ✅ TCP connection to master {master_addr}:{master_port} succeeded.")
                return
        except Exception as e:
            print(f"[rank{rank}] Waiting for master... attempt {attempt+1}, error: {e}")
            time.sleep(2)

    print(f"[rank{rank}] ❌ TCP connection to master {master_addr}:{master_port} failed after 10 attempts.")
    exit(1)

def test_internode_connectivity():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    # Allocate a tensor and run all_reduce to check communication
    tensor = torch.tensor([rank], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] All-reduce test result: {tensor.item()} (should equal sum 0..{world_size - 1})")
    print(f"[Rank {dist.get_rank()}] World size: {dist.get_world_size()}")
    dist.barrier()  # Optional: sync all ranks after test
    time.sleep(1)   # Avoid race conditions in stdout

def broadcast_config(config, rank):
    if rank == 0:
        obj_list = [config]
    else:
        obj_list = [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]



# At module level
def set_global_context(train_loader, val_loader, test_loader, device, rank, local_rank, distributed):
    global _global_train_loader, _global_val_loader, _global_test_loader
    global _global_device, _global_rank, _global_local_rank, _global_distributed

    _global_train_loader = train_loader
    _global_val_loader = val_loader
    _global_test_loader = test_loader
    _global_device = device
    _global_rank = rank
    _global_local_rank = local_rank
    _global_distributed = distributed


def train(config, train_loader, val_loader, test_loader, device, rank, local_rank, distributed):
    use_wandb = bool(config.get("use_wandb", True))
    wandb_enabled = rank == 0 and use_wandb

    if wandb_enabled:
        wandb.init(project="ai-diffraction", config=config)
        run = wandb.run
        config = wandb.config  # Get hyperparameters from sweep
    else:
        run = None

    run_id = None
    model_dir = None
    final_model_path = None
    latest_checkpoint_path = None
    best_checkpoint_path = None
    best_val_loss = float("inf")
    start_epoch = 0

        
    # Initialize model
    label_mode = get_label_mode(config)
    num_output = get_model_output_dim(config)

    model = VIT_model(
        spec_length=config["spec_length"],
        num_output=num_output,
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        drop_ratio=config["dropout"],
        use_rope=config["use_rope"],
        use_mlp_head=config["use_mlp_head"],
        mlp_head_hidden_dim=config["mlp_head_hidden_dim"],
        use_physics_pe=get_nested_config_value(config, "use_physics_pe", False),
        physics_pe_mode=get_nested_config_value(config, "physics_pe_mode", "sin2theta"),
        two_theta_min=float(get_nested_config_value(config, "two_theta_min", 5.0)),
        two_theta_max=float(get_nested_config_value(config, "two_theta_max", 90.0)),
        physics_pe_scale=float(get_nested_config_value(config, "physics_pe_scale", 1.0)),
        use_coordinate_channel=get_nested_config_value(config, "use_coordinate_channel", False),
        coordinate_mode=get_nested_config_value(config, "coordinate_mode", "sin2theta"),
    )

#    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if label_mode == "multilabel":
        if use_split_head(config):
            model.head_sys = nn.Linear(model.embed_dim, 7)
            model.head_lat = nn.Linear(model.embed_dim, 5)
            model.head_ops = nn.Linear(model.embed_dim, 25)
        if use_aux_ext_head(config):
            model.aux_ext_head = nn.Linear(model.embed_dim, get_nested_config_value(config, "num_classes", 99))
    model = model.to(device)

    # Only trace the model on rank 0 (avoid DDP duplicates)
    if wandb_enabled:
        wandb.watch(model, log="all", log_freq=100)  # Can change to "gradients" if overhead is high


    model_summary_str = ""
    with contextlib.redirect_stdout(io.StringIO()) as f:
        summary(model, input_size=(1, 1, model.spec_length))
        model_summary_str = f.getvalue()

    cleaned_summary = clean_summary_str(model_summary_str)

    if wandb_enabled:
        # Now log to W&B (e.g., as an artifact or a text panel)
        wandb.log({"model_summary": wandb.Html(f"<pre>{cleaned_summary}</pre>")})

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]) if config["optimizer"] == "Adam" else optim.SGD(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    # summary(model, input_size=(config["parameters"]["batch_size"]["value"], 1, config["spec_length"]))

    # --- Scheduler setup ---
    scheduler = None
    if config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["num_epochs"],  # full training period
            eta_min=config.get("eta_min", 1e-6)
        )
    elif config["scheduler"] == "cosine_warm_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get("T_0", 10),     # first restart period
            T_mult=config.get("T_mult", 1), # multiplier for subsequent cycles
            eta_min=config.get("eta_min", 1e-6)
        )
    elif config["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("decay_rate", 0.9)
        )

    warmup_steps = int(get_nested_config_value(config, "warmup_steps", 0) or 0)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    global_step = 0
    if warmup_steps > 0:
        for group in optimizer.param_groups:
            group["lr"] = 0.0

    resume_checkpoint = get_resume_checkpoint(config)
    if resume_checkpoint:
        if rank == 0:
            print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        resume_weights_only = get_resume_weights_only(config)
        model_state_dict = get_checkpoint_model_state_dict(checkpoint)
        original_patch_shape = model_state_dict.get("patch_embed.proj.weight", None)
        original_patch_shape = tuple(original_patch_shape.shape) if original_patch_shape is not None else None
        model_patch_shape = tuple(model.state_dict()["patch_embed.proj.weight"].shape)
        adapted_state_dict = adapt_patch_embed_input_channels(model_state_dict, model)
        model.load_state_dict(adapted_state_dict, strict=False)
        patch_shape_changed = original_patch_shape is not None and original_patch_shape != model_patch_shape
        if (
            isinstance(checkpoint, dict)
            and "optimizer_state_dict" in checkpoint
            and checkpoint["optimizer_state_dict"] is not None
            and not patch_shape_changed
            and not resume_weights_only
        ):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (
            scheduler is not None
            and isinstance(checkpoint, dict)
            and checkpoint.get("scheduler_state_dict") is not None
            and not patch_shape_changed
            and not resume_weights_only
        ):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        elif rank == 0 and (patch_shape_changed or resume_weights_only):
            reason = (
                "resume_weights_only is enabled"
                if resume_weights_only
                else f"patch_embed.proj.weight changed shape from {original_patch_shape} to {model_patch_shape}"
            )
            print(f"Skipping optimizer/scheduler resume because {reason}")
        if isinstance(checkpoint, dict) and not resume_weights_only:
            start_epoch = int(checkpoint.get("epoch", 0))
            best_val_loss = float(checkpoint.get("val_loss", float("inf")))
            global_step = int(checkpoint.get("global_step", start_epoch * len(train_loader)))
        else:
            start_epoch = 0
            best_val_loss = float("inf")
            global_step = 0

    template_bank = None
    ext_group_order = None
    templates = None
    if label_mode == "multilabel":
        template_bank, ext_group_order, templates = build_template_bank(
            canonical_table_path=get_nested_config_value(config, "canonical_table_path", None),
            final_table_path=get_nested_config_value(config, "final_table_path", None),
            sg_lookup_path=get_nested_config_value(config, "sg_lookup_path", None),
        )
    # Loss function
    criterion = nn.BCEWithLogitsLoss() if label_mode == "multilabel" else nn.CrossEntropyLoss()
    aux_criterion = nn.CrossEntropyLoss() if label_mode == "multilabel" and use_aux_ext_head(config) else None
    split_criteria = None
    if label_mode == "multilabel" and use_split_head(config):
        split_criteria = {
            "system": nn.CrossEntropyLoss(),
            "lattice": nn.CrossEntropyLoss(),
            "operators": build_operator_loss(config, templates, device, rank),
        }

    if rank == 0:
        run_id = wandb.run.id if wandb_enabled else f"local_{int(time.time())}"
        model_dir = config["model_path"]
        os.makedirs(model_dir, exist_ok=True)
        final_model_path = os.path.join(model_dir, f"xrd_model_{run_id}.pth")
        latest_checkpoint_path = os.path.join(model_dir, f"xrd_model_{run_id}_latest.pth")
        best_checkpoint_path = os.path.join(model_dir, f"xrd_model_{run_id}_best.pth")

    # Training loop
    for epoch in range(start_epoch, config["num_epochs"]):
        train_sampler = getattr(train_loader, "sampler", None)
        if distributed and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config['num_epochs']}",
            disable=(rank != 0),
        )
        for i, batch in enumerate(progress_bar):
            inputs, targets, ext_targets = unpack_batch(batch, label_mode)
            inputs = inputs.to(device)
            targets = targets.to(device)
            ext_targets = ext_targets.to(device)

            optimizer.zero_grad()
            outputs, aux_outputs = forward_model_outputs(model, inputs, config)
            loss = compute_primary_loss(outputs, targets, label_mode, criterion, config, split_criteria=split_criteria)
            if aux_criterion is not None:
                loss = loss + get_aux_ext_weight(config) * aux_criterion(aux_outputs, ext_targets)
            loss.backward()
            optimizer.step()
            global_step += 1

            if warmup_steps > 0 and global_step <= warmup_steps:
                warmup_scale = global_step / warmup_steps
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * warmup_scale
            elif warmup_steps > 0 and global_step == warmup_steps + 1:
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr

            # Step scheduler
            if scheduler is not None:
                if config["scheduler"] == "cosine_warm_restart" and global_step > warmup_steps:
                    scheduler.step(epoch + i / len(train_loader))


            running_loss += loss.item()

        running_loss_tensor = torch.tensor(running_loss, device=device)
        if distributed:
            dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = running_loss_tensor.item() / (len(train_loader) * (dist.get_world_size() if distributed else 1))
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {avg_train_loss}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        val_aux_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, ext_targets = unpack_batch(batch, label_mode)
                inputs = inputs.to(device)
                targets = targets.to(device)
                ext_targets = ext_targets.to(device)

                outputs, aux_outputs = forward_model_outputs(model, inputs, config)
                loss = compute_primary_loss(outputs, targets, label_mode, criterion, config, split_criteria=split_criteria)
                if aux_criterion is not None:
                    loss = loss + get_aux_ext_weight(config) * aux_criterion(aux_outputs, ext_targets)
                val_loss += loss.item()

                if label_mode == "multilabel":
                    if use_split_head(config):
                        pred_ext = decode_split_head_logits(outputs, template_bank, ext_group_order)
                    else:
                        pred_ext = decode_multilabel_logits(outputs, template_bank, ext_group_order)
                    val_correct += (pred_ext == (ext_targets + 1)).sum().item()
                    val_total += ext_targets.size(0)
                    if aux_outputs is not None:
                        val_aux_correct += (torch.argmax(aux_outputs, dim=1) == ext_targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        if distributed:
            val_stats = torch.tensor(
                [val_loss, float(val_correct), float(val_total), float(val_aux_correct)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
            avg_val_loss = val_stats[0].item() / (len(val_loader) * dist.get_world_size())
            val_correct = int(val_stats[1].item())
            val_total = int(val_stats[2].item())
            val_aux_correct = int(val_stats[3].item())

        # --- Step epoch-based schedulers (after all batches in the epoch) ---
        if scheduler is not None:
            if config["scheduler"] != "cosine_warm_restart":
                # Already stepped per batch
 #               current_lr = optimizer.param_groups[0]["lr"]
#            else:
                # Step epoch-based schedulers like StepLR or CosineAnnealingLR
                scheduler.step()
#                current_lr = optimizer.param_groups[0]["lr"]

        current_lr = optimizer.param_groups[0]["lr"]

        if rank == 0:
            print(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {avg_train_loss}")
            print(f"Epoch {epoch + 1}/{config['num_epochs']}, Validation Loss: {avg_val_loss}")
            if label_mode == "multilabel" and val_total > 0:
                print(f"Epoch {epoch + 1}/{config['num_epochs']}, Validation Decode Accuracy: {100 * val_correct / val_total:.2f}%")
                if aux_criterion is not None:
                    print(f"Epoch {epoch + 1}/{config['num_epochs']}, Validation Aux EG Accuracy: {100 * val_aux_correct / val_total:.2f}%")

        if wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
                **({"val_decode_accuracy": 100 * val_correct / val_total} if label_mode == "multilabel" and val_total > 0 else {}),
                **({"val_aux_ext_accuracy": 100 * val_aux_correct / val_total} if aux_criterion is not None and val_total > 0 else {})
            })

        if rank == 0:
            model_state = model.module.state_dict() if distributed else model.state_dict()
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "aux_ext_head_state_dict": (model.module.aux_ext_head.state_dict() if distributed and aux_criterion is not None else model.aux_ext_head.state_dict() if aux_criterion is not None else None),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "config": dict(config),
                "run_id": run_id,
                "global_step": global_step,
            }

            torch.save(checkpoint, latest_checkpoint_path)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(checkpoint, best_checkpoint_path)

    if rank == 0:
        # Save model after training
        model_state = model.module.state_dict() if distributed else model.state_dict()
        torch.save(model_state, final_model_path)
        if wandb_enabled:
            wandb.save(final_model_path)
            wandb.save(latest_checkpoint_path)
            wandb.save(best_checkpoint_path)

        print("After model save")
        print("After embedding logging")

    # Evaluation
    model.eval()
    test_loss, correct, total = 0.0, 0.0, 0.0
    top1_correct, top3_correct, top5_correct = 0.0, 0.0, 0.0
    aux_top1_correct = 0.0
    aux_top3_correct = 0.0
    aux_top5_correct = 0.0

    local_preds = []
    local_targets = []
    local_top5_preds = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, ext_targets = unpack_batch(batch, label_mode)
            inputs = inputs.to(device)
            targets = targets.to(device)
            ext_targets = ext_targets.to(device)

            outputs, aux_outputs = forward_model_outputs(model, inputs, config)
            loss = compute_primary_loss(outputs, targets, label_mode, criterion, config, split_criteria=split_criteria)
            if aux_criterion is not None:
                loss = loss + get_aux_ext_weight(config) * aux_criterion(aux_outputs, ext_targets)
            test_loss += loss.item()

            if label_mode == "multilabel":
                if use_split_head(config):
                    predicted = decode_split_head_logits(outputs, template_bank, ext_group_order) - 1
                else:
                    predicted = decode_multilabel_logits(outputs, template_bank, ext_group_order) - 1
            else:
                _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == ext_targets).sum().item()

            if label_mode == "multilabel":
                if use_split_head(config):
                    top1_correct += compute_split_head_topk_accuracy(outputs, ext_targets, template_bank, ext_group_order, k=1)
                    top3_correct += compute_split_head_topk_accuracy(outputs, ext_targets, template_bank, ext_group_order, k=3)
                    top5_correct += compute_split_head_topk_accuracy(outputs, ext_targets, template_bank, ext_group_order, k=5)
                    top5_preds = topk_decoded_split_head_ext_groups(outputs, template_bank, ext_group_order, k=5).cpu().numpy() - 1
                else:
                    top1_correct += compute_multilabel_topk_accuracy(outputs, ext_targets, template_bank, ext_group_order, k=1)
                    top3_correct += compute_multilabel_topk_accuracy(outputs, ext_targets, template_bank, ext_group_order, k=3)
                    top5_correct += compute_multilabel_topk_accuracy(outputs, ext_targets, template_bank, ext_group_order, k=5)
                    top5_preds = topk_decoded_ext_groups(outputs, template_bank, ext_group_order, k=5).cpu().numpy() - 1
                if aux_outputs is not None:
                    aux_top1_correct += compute_top_k_accuracy(aux_outputs, ext_targets, k=1)
                    aux_top3_correct += compute_top_k_accuracy(aux_outputs, ext_targets, k=3)
                    aux_top5_correct += compute_top_k_accuracy(aux_outputs, ext_targets, k=5)
                targets_np = ext_targets.cpu().numpy()
            else:
                top1_correct += compute_top_k_accuracy(outputs, targets, k=1)
                top3_correct += compute_top_k_accuracy(outputs, targets, k=3)
                top5_correct += compute_top_k_accuracy(outputs, targets, k=5)
                top5_preds = torch.topk(outputs, k=5, dim=1).indices.cpu().numpy()
                targets_np = targets.cpu().numpy()

            local_preds.append(predicted.cpu().numpy())
            local_targets.append(ext_targets.cpu().numpy())

            top5_selected_preds = []
            for i in range(targets_np.shape[0]):
                if targets_np[i] in top5_preds[i]:
                    top5_selected_preds.append(targets_np[i])
                else:
                    top5_selected_preds.append(top5_preds[i][0])
            local_top5_preds.append(np.array(top5_selected_preds))

    print("After test")

    metric_stats = torch.tensor(
        [test_loss, correct, total, top1_correct, top3_correct, top5_correct, aux_top1_correct, aux_top3_correct, aux_top5_correct],
        device=device,
        dtype=torch.float64,
    )
    if distributed:
        dist.all_reduce(metric_stats, op=dist.ReduceOp.SUM)

    test_loss = metric_stats[0].item()
    correct = metric_stats[1].item()
    total = metric_stats[2].item()
    top1_correct = metric_stats[3].item()
    top3_correct = metric_stats[4].item()
    top5_correct = metric_stats[5].item()
    aux_top1_correct = metric_stats[6].item()
    aux_top3_correct = metric_stats[7].item()
    aux_top5_correct = metric_stats[8].item()

    test_accuracy = None
    top1_accuracy = None
    top3_accuracy = None
    top5_accuracy = None
    aux_top1_accuracy = None
    aux_top3_accuracy = None
    aux_top5_accuracy = None

    if rank == 0:
        print(f"Total samples: {int(total)}")
        if total > 0:
            test_accuracy = 100 * correct / total
            top1_accuracy = 100 * top1_correct / total
            top3_accuracy = 100 * top3_correct / total
            top5_accuracy = 100 * top5_correct / total
            aux_top1_accuracy = 100 * aux_top1_correct / total if aux_criterion is not None else None
            aux_top3_accuracy = 100 * aux_top3_correct / total if aux_criterion is not None else None
            aux_top5_accuracy = 100 * aux_top5_correct / total if aux_criterion is not None else None

            print(f"Test Accuracy: {test_accuracy:.2f}%")
            print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
            print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")
            print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
            if aux_criterion is not None:
                print(f"Aux EG Top-1 Accuracy: {aux_top1_accuracy:.2f}%")
                print(f"Aux EG Top-3 Accuracy: {aux_top3_accuracy:.2f}%")
                print(f"Aux EG Top-5 Accuracy: {aux_top5_accuracy:.2f}%")
        else:
            print("No test samples present; skipping test metrics.")

    gathered_preds = [None] * dist.get_world_size() if distributed and rank == 0 else None
    gathered_targets = [None] * dist.get_world_size() if distributed and rank == 0 else None
    gathered_top5_preds = [None] * dist.get_world_size() if distributed and rank == 0 else None

    if distributed:
        pred_payload = np.concatenate(local_preds) if local_preds else np.array([], dtype=np.int64)
        target_payload = np.concatenate(local_targets) if local_targets else np.array([], dtype=np.int64)
        top5_payload = np.concatenate(local_top5_preds) if local_top5_preds else np.array([], dtype=np.int64)
        dist.gather_object(pred_payload, gathered_preds, dst=0)
        dist.gather_object(target_payload, gathered_targets, dst=0)
        dist.gather_object(top5_payload, gathered_top5_preds, dst=0)
    else:
        gathered_preds = [np.concatenate(local_preds)] if local_preds else [np.array([], dtype=np.int64)]
        gathered_targets = [np.concatenate(local_targets)] if local_targets else [np.array([], dtype=np.int64)]
        gathered_top5_preds = [np.concatenate(local_top5_preds)] if local_top5_preds else [np.array([], dtype=np.int64)]

    if rank == 0:
        all_preds = np.concatenate([arr for arr in gathered_preds if arr.size > 0]) if gathered_preds else np.array([], dtype=np.int64)
        all_targets = np.concatenate([arr for arr in gathered_targets if arr.size > 0]) if gathered_targets else np.array([], dtype=np.int64)
        all_top5_preds = np.concatenate([arr for arr in gathered_top5_preds if arr.size > 0]) if gathered_top5_preds else np.array([], dtype=np.int64)

        if wandb_enabled and total > 0:
            wandb.log({
                "total_samples": int(total),
                "test_loss": test_loss,
                "test_accuracy_top1": top1_accuracy,
                "test_accuracy_top3": top3_accuracy,
                "test_accuracy_top5": top5_accuracy,
                **({"aux_test_accuracy_top1": aux_top1_accuracy, "aux_test_accuracy_top3": aux_top3_accuracy, "aux_test_accuracy_top5": aux_top5_accuracy} if aux_criterion is not None else {}),
            })

        if wandb_enabled and all_targets.size > 0 and all_preds.size > 0:
            wandb.log({
                f"conf_mat_top1_{wandb.run.name}": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_targets,
                    preds=all_preds,
                    class_names=[str(i) for i in range(get_nested_config_value(config, "num_classes", 99))]
                )
            })

        # ---- Log CLS embeddings ----
#        cls_embeddings = torch.cat(cls_embeddings, dim=0)
#        cls_labels = torch.cat(cls_labels, dim=0)
#        embedding_dim = cls_embeddings.shape[1]

#        columns = ["label"] + [f"D{i}" for i in range(embedding_dim)]
#        data = []
#        for i in range(cls_embeddings.shape[0]):
#            row = [str(cls_labels[i].item())] + list(cls_embeddings[i].numpy())
#            data.append(row)

#        cls_table = wandb.Table(columns=columns, data=data)
#        wandb.log({"cls_embedding_projector": cls_table})

    if wandb_enabled:
        wandb.finish()

# def run_sweep(sweep_id, config, train_loader, val_loader, test_loader, device, num_sweeps, rank, local_rank, distributed):
#     print(f"Using Sweep ID: {sweep_id}")
#     #sweep_id = wandb.sweep(config, project="XRD_ViT_Code_sweep")
#     #wandb.init(project="ai-diffraction", config=config)
    
#     # Run agent only on rank 0 to avoid launching multiple concurrent runs
#     if rank == 0:
#         wandb.agent(sweep_id, function=lambda: train(config, train_loader, val_loader, test_loader, device, rank, local_rank, distributed), count=num_sweeps, project="ai-diffraction")
#     else:
#         # Other ranks wait to receive config and run training
#         for _ in range(num_sweeps):
#             sweep_config = broadcast_config(None, rank)
#             train(
#                 sweep_config,
#                 train_loader,
#                 val_loader,
#                 test_loader,
#                 device,
#                 rank,
#                 local_rank,
#                 distributed
#             )

def run_sweep(sweep_id, config, train_loader, val_loader, test_loader, device,
              num_sweeps, rank, local_rank, distributed):
    print(f"Using Sweep ID: {sweep_id}")

    if rank == 0:
        # This lambda keeps your train_entry_point exactly as-is, just partially applied
        wandb.agent(
            sweep_id,
            function=lambda: train_entry_point(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                rank=rank,
                local_rank=local_rank,
                distributed=distributed
            ),
            count=num_sweeps,
            project="ai-diffraction"
        )
    else:
        # All other ranks receive the config via broadcast and run training
        for _ in range(num_sweeps):
            broadcasted_config = broadcast_config(None, rank)
            train(
                broadcasted_config,
                train_loader,
                val_loader,
                test_loader,
                device,
                rank,
                local_rank,
                distributed
            )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def train_entry_point(config=None, train_loader=None, val_loader=None, test_loader=None,
                      device=None, rank=None, local_rank=None, distributed=None):
    run = wandb.init()  # ✅ Must happen first

    if config is None:
        config = dict(wandb.config)  # Only fetch from wandb if not passed explicitly

    # Broadcast to other ranks
    broadcasted_config = broadcast_config(dict(config), rank)


    # Run training with the provided config and data loaders
    train(
        config,
        train_loader,
        val_loader,
        test_loader,
        device,
        rank,
        local_rank,
        distributed
    )

def main():
    args = parse_args()
    config = load_config(args.config)
    config["use_wandb"] = not args.disable_wandb

    # Global rank
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    print("local rank: " + str(local_rank))

    # Wait for master TCP port (if not rank 0)
    wait_for_master()

    if args.distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
        local_rank = 0

    # sanity test of DDP connectivity
    if args.distributed:
        test_internode_connectivity()

    # # Infer num_workers based on CPU cores per process
    # total_cores = os.cpu_count() or 16  # fallback to 16 if None
    # max_workers = 16
    # num_workers = min(max_workers, max(1, total_cores // world_size))

    # Data loading
    data_mode = get_nested_config_value(config, "data_mode", "hdf5")
    batch_size = get_nested_config_value(config, "batch_size")

    if data_mode == "streaming":
        stream_cfg = StreamingConfig(
            samples_per_epoch=get_nested_config_value(config, "samples_per_epoch", 10000),
            batch_size=batch_size,
            num_workers=get_nested_config_value(config, "num_workers", 0),
            twotheta_min=get_nested_config_value(config, "two_theta_min", 10.0),
            twotheta_max=get_nested_config_value(config, "two_theta_max", 110.0),
            step_size=get_nested_config_value(config, "step_size_stream", 0.1),
            weighted_sampling=get_nested_config_value(config, "weighted_sampling", False),
            label_mode=get_label_mode(config),
            final_table_path=get_nested_config_value(config, "final_table_path", None) or None,
            canonical_table_path=get_nested_config_value(config, "canonical_table_path", None),
            sg_lookup_path=get_nested_config_value(config, "sg_lookup_path", None),
        )
        train_loader = get_streaming_dataloader(stream_cfg)
        val_loader = None
        test_loader = None
        data_path = get_nested_config_value(config, "data_path", None)
        if data_path:
            _, val_loader, test_loader, num_classes, intensity_points = get_dataloaders(
                data_path,
                batch_size=batch_size,
                world_size=world_size,
                num_workers=get_nested_config_value(config, "num_workers", 0),
                num_classes=get_nested_config_value(config, "num_classes", 99),
                prefetch_factor=get_nested_config_value(config, "prefetch_factor", 2),
                start_col=get_nested_config_value(config, "start_col", 1),
                end_col=get_nested_config_value(config, "end_col", get_nested_config_value(config, "spec_length", 8501)),
                label_mode=get_label_mode(config),
                canonical_table_path=get_nested_config_value(config, "canonical_table_path", None),
                final_table_path=get_nested_config_value(config, "final_table_path", None),
                sg_lookup_path=get_nested_config_value(config, "sg_lookup_path", None),
                max_samples_val=get_nested_config_value(config, "max_samples_val", None),
                max_samples_test=get_nested_config_value(config, "max_samples_test", None),
            )
        else:
            num_classes = get_nested_config_value(config, "num_classes", 99)
            intensity_points = stream_cfg.num_points
            val_loader = train_loader
            test_loader = train_loader
    elif data_mode == "mixed_hdf5":
        train_loader, val_loader, test_loader, num_classes, intensity_points = get_mixed_dataloaders(
            get_nested_config_value(config, "standard_data_path"),
            get_nested_config_value(config, "po_data_path"),
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            num_workers=get_nested_config_value(config, "num_workers", 0),
            num_classes=get_nested_config_value(config, "num_classes", 99),
            prefetch_factor=get_nested_config_value(config, "prefetch_factor", 2),
            start_col=get_nested_config_value(config, "start_col", 1),
            end_col=get_nested_config_value(config, "end_col", get_nested_config_value(config, "spec_length", 8501)),
            label_mode=get_label_mode(config),
            canonical_table_path=get_nested_config_value(config, "canonical_table_path", None),
            final_table_path=get_nested_config_value(config, "final_table_path", None),
            sg_lookup_path=get_nested_config_value(config, "sg_lookup_path", None),
            max_samples_train=get_nested_config_value(config, "max_samples_train", None),
            max_samples_val=get_nested_config_value(config, "max_samples_val", None),
            max_samples_test=get_nested_config_value(config, "max_samples_test", None),
            po_train_ratio=get_nested_config_value(config, "po_train_ratio", 0.2),
            po_val_ratio=get_nested_config_value(config, "po_val_ratio", 0.2),
            po_test_ratio=get_nested_config_value(config, "po_test_ratio", 0.2),
            mixed_seed=get_nested_config_value(config, "mixed_seed", 1337),
        )
    else:
        data_path = get_nested_config_value(config, "data_path")
        train_loader, val_loader, test_loader, num_classes, intensity_points = get_dataloaders(
            data_path,
            batch_size=batch_size,
            world_size=world_size,
            num_workers=get_nested_config_value(config, "num_workers", 0),
            num_classes=get_nested_config_value(config, "num_classes", 99),
            prefetch_factor=get_nested_config_value(config, "prefetch_factor", 2),
            start_col=get_nested_config_value(config, "start_col", 1),
            end_col=get_nested_config_value(config, "end_col", get_nested_config_value(config, "spec_length", 8501)),
            label_mode=get_label_mode(config),
            canonical_table_path=get_nested_config_value(config, "canonical_table_path", None),
            final_table_path=get_nested_config_value(config, "final_table_path", None),
            sg_lookup_path=get_nested_config_value(config, "sg_lookup_path", None),
            max_samples_train=get_nested_config_value(config, "max_samples_train", None),
            max_samples_val=get_nested_config_value(config, "max_samples_val", None),
            max_samples_test=get_nested_config_value(config, "max_samples_test", None),
        )

    if args.sweep:
        if not args.sweep_id:
            raise ValueError("--sweep requires --sweep_id")
        run_sweep(args.sweep_id, config, train_loader, val_loader, test_loader, device, args.num_sweeps, rank, local_rank, args.distributed)
    else:
        train(config, train_loader, val_loader, test_loader, device, rank, local_rank, args.distributed)


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
