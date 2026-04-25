import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset, Sampler


# =========================
# HDF5 Dataset (SEQUENTIAL)
# =========================
class H5Dataset(Dataset):
    def __init__(self, h5_path, split):
        self.h5_path = h5_path
        self.split = split
        self.file = None

        with h5py.File(self.h5_path, "r") as f:
            self.length = len(f[f"X_{split}"])

    def _get_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r", swmr=True)
        return self.file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self._get_file()
        x = f[f"X_{self.split}"][idx]
        y = f[f"y_{self.split}"][idx] - 1
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )


# =========================
# Chunk-aware shuffling sampler
# Reads data in large sequential chunks, shuffles within each chunk
# Avoids random seeks across the full dataset
# =========================
class ChunkShuffleSampler(Sampler):
    def __init__(self, dataset_size, chunk_size=50000, rank=0, world_size=1, seed=0):
        self.dataset_size = dataset_size
        self.chunk_size = chunk_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        # Split dataset into chunks, shuffle chunk order
        chunk_starts = np.arange(0, self.dataset_size, self.chunk_size)
        rng.shuffle(chunk_starts)

        indices = []
        for start in chunk_starts:
            end = min(start + self.chunk_size, self.dataset_size)
            chunk_indices = np.arange(start, end)
            rng.shuffle(chunk_indices)  # shuffle within chunk
            indices.extend(chunk_indices.tolist())

        # Distribute across ranks
        indices = indices[self.rank::self.world_size]
        return iter(indices)

    def __len__(self):
        return self.dataset_size // self.world_size


# =========================
# DATALOADER FACTORY
# =========================
def get_dataloaders(
    h5_file="xrd_data.h5",
    batch_size=32,
    world_size=1,
    rank=0,
    num_workers=8,
    num_classes=230,
    prefetch_factor=4,
    distributed=True,
    subset_fraction=None,
):
    train_dataset = H5Dataset(h5_file, "train")
    val_dataset   = H5Dataset(h5_file, "val")
    test_dataset  = H5Dataset(h5_file, "test")

    if subset_fraction is not None and 0 < subset_fraction < 1:
        subset_size = int(len(train_dataset) * subset_fraction)
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, indices)

    # Use chunk-aware sampler for train, standard for val/test
    train_sampler = ChunkShuffleSampler(
        dataset_size=len(train_dataset),
        chunk_size=50000,
        rank=rank,
        world_size=world_size,
        seed=42,
    )

    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed and world_size > 1 else None
    )
    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed and world_size > 1 else None
    )

    common_loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=False,  # sampler handles shuffling
        drop_last=True,
        **common_loader_args,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=False,
        **common_loader_args,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        shuffle=False,
        **common_loader_args,
    )

    with h5py.File(h5_file, "r") as f:
        intensity_points = f["X_train"].shape[1]

    return train_loader, val_loader, test_loader, num_classes, intensity_points
