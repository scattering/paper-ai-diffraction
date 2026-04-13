import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data import TensorDataset

from extinction_multilabel import build_extinction_templates, ext_group_to_multilabel_target

# # ====== Custom Dataset Loader ======
# def get_dataloaders(h5_file="xrd_data.h5", batch_size=32, world_size=1, rank=0, num_workers=8, num_classes=230, prefetch_factor=4):
#     """
#     Returns DataLoaders for train, val, and test sets.

#     Args:
#         h5_file (str): Path to the HDF5 dataset.
#         batch_size (int): Batch size per GPU.
#         world_size (int): Total number of GPUs.
#         rank (int): Current process rank (for DDP).

#     Returns:
#         train_loader, val_loader, test_loader, num_classes, intensity_points
#     """

#     # Load dataset from the HDF5 file
#     with h5py.File(h5_file, "r") as f:
#         X_train, y_train = f["X_train"][:], f["y_train"][:] - 1  # Adjust labels to start from 0
#         X_val, y_val = f["X_val"][:], f["y_val"][:] - 1  # Adjust labels to start from 0
#         X_test, y_test = f["X_test"][:], f["y_test"][:] - 1  # Adjust labels to start from 0

#     # Convert to PyTorch tensors
#     X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
#     X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
#     X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

#     # Get the number of classes (assuming labels are categorical)
#     #num_classes = len(torch.unique(y_train))
#     num_classes = num_classes

#     # Get the intensity points (assumed to be the feature length)
#     intensity_points = X_train.shape[1]

#     # Use DistributedSampler for multi-GPU training
#     train_sampler = DistributedSampler(TensorDataset(X_train, y_train), num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
#     val_sampler = DistributedSampler(TensorDataset(X_val, y_val), num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
#     test_sampler = DistributedSampler(TensorDataset(X_test, y_test), num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

#     # Create DataLoaders
#     train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=num_workers, pin_memory=False, prefetch_factor=prefetch_factor, persistent_workers=True, drop_last=True)
#     val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor, persistent_workers=True, pin_memory=False)
#     test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor, persistent_workers=True, pin_memory=False)

#     return train_loader, val_loader, test_loader, num_classes, intensity_points


# ====== Lazy-loading Dataset with chunked batch reads ======
class H5ChunkedDataset(Dataset):
    def __init__(
        self,
        h5_path,
        split,
        start_col,
        end_col,
        max_samples=None,
        label_mode="categorical",
        canonical_table_path=None,
        final_table_path=None,
        sg_lookup_path=None,
    ):
        self.h5_path = h5_path
        self.split = split
        self.file = None  # will be opened once per worker
        with h5py.File(h5_path, "r") as f:
            full_len = f[f"X_{split}"].shape[0]
            self.len = min(max_samples, full_len) if max_samples is not None else full_len

        # Convert to 0-based slice indices
        self.start_col = start_col - 1
        self.end_col = end_col  # Python slice end is exclusive
        self.label_mode = label_mode
        self.templates = None

        if self.label_mode == "multilabel":
            self.templates = build_extinction_templates(
                canonical_table_path=canonical_table_path or None,
                final_table_path=final_table_path or None,
                sg_lookup_path=sg_lookup_path or None,
            )

    def __len__(self):
        return self.len

#    def __getitem__(self, idx):
#        if self.file is None:
#            self.file = h5py.File(self.h5_path, "r")
#
#        X_ds = self.file[f"X_{self.split}"]
#        y_ds = self.file[f"y_{self.split}"]
#
#        # If idx is a list/tensor (batch mode), fetch all at once
#        if torch.is_tensor(idx):
#            idx = idx.tolist()
#        if isinstance(idx, list):
#            X = X_ds[idx, self.start_col:self.end_col]
#            y = y_ds[idx] - 1
#        else:  # single sample
#            X = X_ds[idx, self.start_col:self.end_col]
#            y = y_ds[idx] - 1
#
#        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        try:
            if self.file is None:
                self.file = h5py.File(self.h5_path, "r")

            X_ds = self.file[f"X_{self.split}"]
            y_ds = self.file[f"y_{self.split}"]

            if torch.is_tensor(idx):
                idx = idx.tolist()
            if isinstance(idx, list):
                X = X_ds[idx, self.start_col:self.end_col]
                y = y_ds[idx]
            else:  # single sample
                X = X_ds[idx, self.start_col:self.end_col]
                y = y_ds[idx]

            if isinstance(idx, list):
                inputs = torch.tensor(X, dtype=torch.float32)
                raw_targets = torch.tensor(y, dtype=torch.long)
                if self.label_mode == "multilabel":
                    binary_targets = torch.stack(
                        [ext_group_to_multilabel_target(int(ext), self.templates) for ext in raw_targets.tolist()],
                        dim=0,
                    )
                    return inputs, binary_targets, raw_targets - 1
                return inputs, raw_targets - 1

            input_tensor = torch.tensor(X, dtype=torch.float32)
            raw_target = int(y)
            if self.label_mode == "multilabel":
                binary_target = ext_group_to_multilabel_target(raw_target, self.templates)
                return input_tensor, binary_target, torch.tensor(raw_target - 1, dtype=torch.long)
            return input_tensor, torch.tensor(raw_target - 1, dtype=torch.long)

        except Exception as e:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"[Rank {rank}] Error in __getitem__ for idx={idx}: {e}")
            raise  # re-raise so DDP will fail gracefully


class MixedH5ChunkedDataset(Dataset):
    def __init__(
        self,
        standard_h5_path,
        po_h5_path,
        split,
        start_col,
        end_col,
        max_samples=None,
        po_ratio=None,
        seed=1337,
        label_mode="categorical",
        canonical_table_path=None,
        final_table_path=None,
        sg_lookup_path=None,
    ):
        self.standard_h5_path = standard_h5_path
        self.po_h5_path = po_h5_path
        self.split = split
        self.standard_file = None
        self.po_file = None
        self.start_col = start_col - 1
        self.end_col = end_col
        self.label_mode = label_mode
        self.templates = None

        if self.label_mode == "multilabel":
            self.templates = build_extinction_templates(
                canonical_table_path=canonical_table_path or None,
                final_table_path=final_table_path or None,
                sg_lookup_path=sg_lookup_path or None,
            )

        with h5py.File(standard_h5_path, "r") as f:
            standard_len = f[f"X_{split}"].shape[0]
        with h5py.File(po_h5_path, "r") as f:
            po_len = f[f"X_{split}"].shape[0]

        available_total = standard_len + po_len
        total_len = min(max_samples, available_total) if max_samples is not None else available_total

        if po_ratio is None:
            po_count = min(po_len, total_len // 2 if po_len < total_len else po_len)
            std_count = total_len - po_count
            if std_count > standard_len:
                std_count = standard_len
                po_count = min(po_len, total_len - std_count)
        else:
            po_count = int(round(total_len * float(po_ratio)))
            po_count = min(po_count, po_len)
            std_count = total_len - po_count
            if std_count > standard_len:
                std_count = standard_len
                po_count = min(po_len, total_len - std_count)

        split_offsets = {"train": 0, "val": 1, "test": 2}
        rng = np.random.default_rng(seed + split_offsets.get(split, 0))
        std_indices = rng.choice(standard_len, size=std_count, replace=False) if std_count > 0 else np.empty((0,), dtype=np.int64)
        po_indices = rng.choice(po_len, size=po_count, replace=False) if po_count > 0 else np.empty((0,), dtype=np.int64)

        mixed = [("standard", int(i)) for i in std_indices.tolist()] + [("po", int(i)) for i in po_indices.tolist()]
        rng.shuffle(mixed)
        self.index_map = mixed
        self.len = len(self.index_map)

    def __len__(self):
        return self.len

    def _ensure_files(self):
        if self.standard_file is None:
            self.standard_file = h5py.File(self.standard_h5_path, "r")
        if self.po_file is None:
            self.po_file = h5py.File(self.po_h5_path, "r")

    def __getitem__(self, idx):
        try:
            self._ensure_files()
            source_name, source_idx = self.index_map[idx]
            h5 = self.standard_file if source_name == "standard" else self.po_file
            X_ds = h5[f"X_{self.split}"]
            y_ds = h5[f"y_{self.split}"]

            X = X_ds[source_idx, self.start_col:self.end_col]
            y = int(y_ds[source_idx])

            input_tensor = torch.tensor(X, dtype=torch.float32)
            if self.label_mode == "multilabel":
                binary_target = ext_group_to_multilabel_target(y, self.templates)
                return input_tensor, binary_target, torch.tensor(y - 1, dtype=torch.long)
            return input_tensor, torch.tensor(y - 1, dtype=torch.long)
        except Exception as e:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"[Rank {rank}] Error in MixedH5ChunkedDataset.__getitem__ for idx={idx}: {e}")
            raise


# ====== Worker init to keep file open per worker ======
def _init_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if isinstance(dataset, H5ChunkedDataset):
        dataset.file = h5py.File(dataset.h5_path, "r")
    elif isinstance(dataset, MixedH5ChunkedDataset):
        dataset.standard_file = h5py.File(dataset.standard_h5_path, "r")
        dataset.po_file = h5py.File(dataset.po_h5_path, "r")


# # ====== Custom collate to handle streaming batches from HDF5 ======
# def h5_collate(batch):
#     # If the dataset returned a whole batch (chunk read), it’s already a tuple of tensors
#     if isinstance(batch[0][0], torch.Tensor) and batch[0][0].ndim > 1:
#         return batch[0]
#     else:
#         # Default PyTorch collate
#         return torch.utils.data._utils.collate.default_collate(batch)
    
def h5_collate(batch):
    """
    Ensures batch is always a tuple (inputs, targets), even for single-sample batches.
    """
    # Separate all inputs and targets
    inputs = [b[0] for b in batch]
    targets = [b[1] for b in batch]

    # Stack tensors
    inputs = torch.stack(inputs, dim=0)
    targets = torch.tensor(targets, dtype=torch.long)

    return inputs, targets

def h5_collate_multilabel(batch):
    inputs = torch.stack([b[0] for b in batch], dim=0)
    multilabel_targets = torch.stack([b[1] for b in batch], dim=0)
    ext_targets = torch.stack([b[2] for b in batch], dim=0)
    return inputs, multilabel_targets, ext_targets

def h5_collate_fn(batch):
    """
    Collate function that supports variable-length 1D spectra.
    Pads each spectrum in the batch to the maximum length in that batch.
    """

    # Separate spectra and labels
    specs = [torch.tensor(b[0], dtype=torch.float32) for b in batch]
    labels = [b[1] for b in batch]

    # Determine max length dynamically
    max_len = max(s.shape[0] for s in specs)

    # Pad spectra to max_len (right padding with zeros)
    padded = []
    for s in specs:
        L = s.shape[0]
        if L < max_len:
            pad = torch.zeros(max_len - L, dtype=torch.float32)
            s = torch.cat([s, pad], dim=0)
        padded.append(s)

    # Stack into batch tensors
    padded = torch.stack(padded, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded, labels

# ====== Loader function ======
def get_dataloaders(h5_file="xrd_data.h5", batch_size=32, world_size=1, rank=0,
                    num_workers=8, num_classes=230, prefetch_factor=4, start_col=1, end_col=1000,
                    label_mode="categorical", canonical_table_path=None, final_table_path=None, sg_lookup_path=None,
                    max_samples_train=None, max_samples_val=None, max_samples_test=None):

    train_dataset = H5ChunkedDataset(h5_file, "train", start_col, end_col, max_samples=max_samples_train, label_mode=label_mode,
                                     canonical_table_path=canonical_table_path, final_table_path=final_table_path,
                                     sg_lookup_path=sg_lookup_path)
    val_dataset = H5ChunkedDataset(h5_file, "val", start_col, end_col, max_samples=max_samples_val, label_mode=label_mode,
                                   canonical_table_path=canonical_table_path, final_table_path=final_table_path,
                                   sg_lookup_path=sg_lookup_path)
    test_dataset = H5ChunkedDataset(h5_file, "test", start_col, end_col, max_samples=max_samples_test, label_mode=label_mode,
                                    canonical_table_path=canonical_table_path, final_table_path=final_table_path,
                                    sg_lookup_path=sg_lookup_path)

    collate_fn = h5_collate_multilabel if label_mode == "multilabel" else h5_collate

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=num_workers,
        pin_memory=True, prefetch_factor=prefetch_factor,
        persistent_workers=True, worker_init_fn=_init_worker,
        collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, prefetch_factor=prefetch_factor,
        persistent_workers=True, worker_init_fn=_init_worker,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, prefetch_factor=prefetch_factor,
        persistent_workers=True, worker_init_fn=_init_worker,
        collate_fn=collate_fn
    )

    # with h5py.File(h5_file, "r") as f:
    #     intensity_points = f["X_train"].shape[1]

    intensity_points = end_col - start_col + 1

    return train_loader, val_loader, test_loader, num_classes, intensity_points


def get_mixed_dataloaders(
    standard_h5_file,
    po_h5_file,
    batch_size=32,
    world_size=1,
    rank=0,
    num_workers=8,
    num_classes=230,
    prefetch_factor=4,
    start_col=1,
    end_col=1000,
    label_mode="categorical",
    canonical_table_path=None,
    final_table_path=None,
    sg_lookup_path=None,
    max_samples_train=None,
    max_samples_val=None,
    max_samples_test=None,
    po_train_ratio=0.2,
    po_val_ratio=0.2,
    po_test_ratio=0.2,
    mixed_seed=1337,
):
    train_dataset = MixedH5ChunkedDataset(
        standard_h5_file, po_h5_file, "train", start_col, end_col,
        max_samples=max_samples_train, po_ratio=po_train_ratio, seed=mixed_seed,
        label_mode=label_mode, canonical_table_path=canonical_table_path,
        final_table_path=final_table_path, sg_lookup_path=sg_lookup_path,
    )
    val_dataset = MixedH5ChunkedDataset(
        standard_h5_file, po_h5_file, "val", start_col, end_col,
        max_samples=max_samples_val, po_ratio=po_val_ratio, seed=mixed_seed + 101,
        label_mode=label_mode, canonical_table_path=canonical_table_path,
        final_table_path=final_table_path, sg_lookup_path=sg_lookup_path,
    )
    test_dataset = MixedH5ChunkedDataset(
        standard_h5_file, po_h5_file, "test", start_col, end_col,
        max_samples=max_samples_test, po_ratio=po_test_ratio, seed=mixed_seed + 202,
        label_mode=label_mode, canonical_table_path=canonical_table_path,
        final_table_path=final_table_path, sg_lookup_path=sg_lookup_path,
    )

    collate_fn = h5_collate_multilabel if label_mode == "multilabel" else h5_collate

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=num_workers,
        pin_memory=True, prefetch_factor=prefetch_factor,
        persistent_workers=True, worker_init_fn=_init_worker,
        collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, prefetch_factor=prefetch_factor,
        persistent_workers=True, worker_init_fn=_init_worker,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, prefetch_factor=prefetch_factor,
        persistent_workers=True, worker_init_fn=_init_worker,
        collate_fn=collate_fn
    )

    intensity_points = end_col - start_col + 1
    return train_loader, val_loader, test_loader, num_classes, intensity_points

# ====== Loader function (test only) ======
def get_dataloaders_test(h5_file="xrd_data.h5", batch_size=32, world_size=1, rank=0,
                         num_workers=8, num_classes=230, prefetch_factor=4, start_col=1, end_col=1000,
                         label_mode="categorical", canonical_table_path=None, final_table_path=None, sg_lookup_path=None,
                         max_samples_test=None):

    test_dataset = H5ChunkedDataset(h5_file, "test", start_col, end_col, max_samples=max_samples_test, label_mode=label_mode,
                                    canonical_table_path=canonical_table_path, final_table_path=final_table_path,
                                    sg_lookup_path=sg_lookup_path)
    collate_fn = h5_collate_multilabel if label_mode == "multilabel" else h5_collate

    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    loader_kwargs = dict(
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["worker_init_fn"] = _init_worker

    test_loader = DataLoader(test_dataset, **loader_kwargs)

    intensity_points = end_col - start_col + 1

    return None, None, test_loader, num_classes, intensity_points
