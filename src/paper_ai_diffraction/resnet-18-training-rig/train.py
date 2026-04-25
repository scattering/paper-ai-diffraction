import os
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import wandb
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from TrainingScripts.resnet_model import ResnetClassifier
from resnet50 import ResNet50
from TrainingScripts.dataset import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--sweep", action="store_true", help="Flag to indicate if this is a sweep run")
    parser.add_argument("--distributed", action="store_true", help="Enable DistributedDataParallel (DDP)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument("--sweep_id", type=str, required=True, help="W&B Sweep ID")
    parser.add_argument("--num_sweeps", type=int, required=True, help="W&B Number of Sweeps")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        raw = json.load(f)

    # Flatten W&B sweep format: {"parameters": {"lr": {"value": 0.001}}}
    # into a simple dict: {"lr": 0.001}
    if "parameters" in raw:
        config = {}
        for key, val in raw["parameters"].items():
            if "value" in val:
                config[key] = val["value"]
            elif "values" in val:
                config[key] = val["values"][0]
            elif "min" in val and "max" in val:
                # Use midpoint as a sensible default when run outside a sweep.
                # During a sweep the agent will overwrite these via wandb.config.
                if val.get("distribution") == "int_uniform":
                    config[key] = (val["min"] + val["max"]) // 2
                else:
                    config[key] = (val["min"] + val["max"]) / 2
            else:
                print(f"Warning: no default found for parameter '{key}', skipping.")
        return config

    return raw  # already a flat config, return as-is


def compute_top_k_accuracy(output, target, k=1):
    """Compute top-k accuracy: fraction of samples where true label is in top-k"""
    with torch.no_grad():
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct_k = correct.any(dim=1).float().sum().item()
        return correct_k


def setup_distributed(local_rank):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    return device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(config, device, local_rank, distributed, world_size):
    early_exit_threshold = 5.0
    early_exit_epoch_window = 3
    low_val_acc_epochs = 0

    rank = dist.get_rank() if distributed else 0

    if rank == 0:
        wandb.init(
            project="final-cnn-results",
            config=config,
            id=os.environ.get("WANDB_RUN_ID"),
            resume="allow",
        )
        # Pull sweep-injected hyperparameters back into config.
        # Only overwrite keys where wandb.config has a real (non-None) value.
        # During a sweep the agent fills these in; during a direct run they
        # stay None so we keep the midpoint defaults from load_config instead.
        config.update({k: v for k, v in dict(wandb.config).items() if v is not None})

    # Broadcast config from rank 0 to all other ranks so every process
    # uses the same (potentially sweep-updated) hyperparameters.
    if distributed:
        config_list = [config]
        dist.broadcast_object_list(config_list, src=0)
        config = config_list[0]

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    config["num_workers"] = max(1, slurm_cpus // world_size)

    print("Loading Data")
    train_loader, val_loader, test_loader, _, _ = get_dataloaders(
        h5_file=config["data_path"],
        batch_size=config["batch_size"],
        world_size=world_size,
        rank=rank,
        num_classes=config["num_classes"],
        num_workers=config["num_workers"],
        prefetch_factor=config["prefetch_factor"],
        subset_fraction=1
    )

    print("Loading Model")
    if config["model"] == "resnet18":
        model = ResnetClassifier(
            input_dim=config["input_dim"],
            res_dims=config["res_dims"],
            res_kernel=config["res_kernel"],
            res_stride=config["res_stride"],
            num_blocks=config["num_blocks"],
            first_kernel_size=config["first_kernel_size"],
            first_stride=config["first_stride"],
            first_pool_kernel_size=config["first_pool_kernel_size"],
            first_pool_stride=config["first_pool_stride"],
            num_classes=config["num_classes"],
        ).to(device)

    elif config["model"] == "Resnet50":
        model = ResNet50(in_channels=1, classes=config["num_classes"]).to(device)

    else:
        raise ValueError(f"Unknown model type: {config['model']}")

    # Watch only on rank 0
    if rank == 0:
        wandb.watch(model, log="all", log_freq=500)

    # Wrap in DDP if needed
    if distributed:
        print("Using DistributedDataParallel")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = (
        optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        if config["optimizer"] == "Adam"
        else optim.SGD(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    )

    criterion = nn.CrossEntropyLoss()
    print("Starting Training!")

    for epoch in range(config["num_epochs"]):
        model.train()

        if distributed and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

        running_loss = 0.0
        train_total = 0
        train_correct = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
            inputs, targets = batch
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        # ── All-reduce training metrics across all ranks before rank 0 logs ──
        if distributed:
            metrics = torch.tensor(
                [running_loss, float(train_correct), float(train_total)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            running_loss, train_correct, train_total = metrics.tolist()

        avg_train_loss = running_loss / (len(train_loader) * world_size)
        train_accuracy = 100 * train_correct / train_total

        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{config['num_epochs']}, "
                f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
            )

            model.eval()
            val_loss = 0.0
            val_total = 0
            val_correct = 0

            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    inputs = inputs.unsqueeze(1)
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)

                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total

            print(
                f"Epoch {epoch + 1}/{config['num_epochs']}, "
                f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
            )

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            })

            # Early stopping check
            if epoch + 1 >= early_exit_epoch_window:
                if val_accuracy < early_exit_threshold:
                    low_val_acc_epochs += 1
                else:
                    low_val_acc_epochs = 0

            if low_val_acc_epochs >= early_exit_epoch_window:
                print(
                    f"[Early Exit] Validation accuracy below threshold for "
                    f"{early_exit_epoch_window} consecutive epochs. Ending run."
                )
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": 9999,
                    "val_loss": 9999,
                    "train_accuracy": 0.0,
                    "val_accuracy": 0.0,
                })
                wandb.finish(exit_code=0)
                return

    print("----------Training Done----------------")

    if rank == 0:
        save_model = model.module if distributed else model
        model_path = f"{config['model_path']}xrd_model_{wandb.run.id}.pth"
        torch.save(save_model.state_dict(), model_path)
        wandb.save(model_path)

        save_model.eval()
        test_loss, correct, total = 0.0, 0, 0
        top1_correct, top3_correct, top5_correct = 0, 0, 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.unsqueeze(1)
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = save_model(inputs)
                loss = criterion(outputs, targets)

                batch_size = targets.size(0)
                test_loss += loss.item() * batch_size
                total += batch_size

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()

                top1_correct += compute_top_k_accuracy(outputs, targets, k=1)
                top3_correct += compute_top_k_accuracy(outputs, targets, k=3)
                top5_correct += compute_top_k_accuracy(outputs, targets, k=5)

        avg_test_loss = test_loss / total
        test_accuracy = 100 * correct / total
        top1_accuracy = 100 * top1_correct / total
        top3_accuracy = 100 * top3_correct / total
        top5_accuracy = 100 * top5_correct / total

        print(f"Total samples: {total}")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
        print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")
        print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

        wandb.log({
            "total_samples": total,
            "test_loss": avg_test_loss,
            "test_accuracy_top1": top1_accuracy,
            "test_accuracy_top3": top3_accuracy,
            "test_accuracy_top5": top5_accuracy,
        })

        wandb.finish()


def main():
    args = parse_args()
    config = load_config(args.config)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Sanity check on rank 0 only — confirm config flattened correctly
    if local_rank == 0:
        print(f"Config keys: {list(config.keys())}")

    distributed = args.distributed

    if distributed:
        device = setup_distributed(local_rank)
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        world_size = 1

    config["num_workers"] = max(1, (os.cpu_count() or 4) // world_size)
    try:
        train(config, device, local_rank, distributed, world_size)
    finally:
        if distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
