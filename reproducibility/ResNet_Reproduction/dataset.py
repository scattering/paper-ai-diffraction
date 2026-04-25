import torch
import h5py
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data import TensorDataset

# ====== Custom Dataset Loader ======
def get_dataloaders(h5_file="xrd_data.h5", batch_size=32, world_size=1, rank=0, num_workers=8, num_classes=230, prefetch_factor=4):
    """
    Returns DataLoaders for train, val, and test sets.

    Args:
        h5_file (str): Path to the HDF5 dataset.
        batch_size (int): Batch size per GPU.
        world_size (int): Total number of GPUs.
        rank (int): Current process rank (for DDP).

    Returns:
        train_loader, val_loader, test_loader, num_classes, intensity_points
    """

    # Load dataset from the HDF5 file
    with h5py.File(h5_file, "r") as f:
        X_train, y_train = f["X_train"][:], f["y_train"][:] - 1  # Adjust labels to start from 0
        X_val, y_val = f["X_val"][:], f["y_val"][:] - 1  # Adjust labels to start from 0
        X_test, y_test = f["X_test"][:], f["y_test"][:] - 1  # Adjust labels to start from 0

    # Convert to PyTorch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

    # Get the number of classes (assuming labels are categorical)
    #num_classes = len(torch.unique(y_train))
    num_classes = num_classes

    # Get the intensity points (assumed to be the feature length)
    intensity_points = X_train.shape[1]

    # Use DistributedSampler for multi-GPU training
    train_sampler = DistributedSampler(TensorDataset(X_train, y_train), num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(TensorDataset(X_val, y_val), num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(TensorDataset(X_test, y_test), num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes, intensity_points
