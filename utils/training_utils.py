"""Distributed training utilities adapted from DiffSinger."""

import numpy as np
import paddle.distributed as dist
from paddle.io import DataLoader, Sampler


class DsBatchSampler(Sampler):
    """Dynamic batch sampler for distributed training.

    Features:
    - Batch by total frames (for consistent GPU memory usage)
    - Sort by similar size (for efficient padding)
    - Distributed sampling across multiple GPUs
    - Optional shuffling at sample and batch levels

    Args:
        dataset: Dataset with `sizes` and `num_frames()` methods.
        max_batch_frames: Maximum total frames per batch.
        max_batch_size: Maximum number of samples per batch.
        num_replicas: Number of distributed processes.
        rank: Rank of current process.
        shuffle_sample: Shuffle samples before batching.
        shuffle_batch: Shuffle batches before distributing.
        sort_by_similar_size: Group similar-length samples for efficiency.
        size_reversed: Reverse sort order (longest first).
        frame_count_grid: Grid size for size-based sorting (default: 5000 frames).
        seed: Random seed.
        drop_last: Drop incomplete batches at the end.
    """

    def __init__(
        self,
        dataset,
        max_batch_frames,
        max_batch_size,
        num_replicas=1,
        rank=0,
        shuffle_sample=True,
        shuffle_batch=True,
        sort_by_similar_size=True,
        size_reversed=False,
        frame_count_grid=5000,
        seed=0,
        drop_last=False,
    ):
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.max_batch_frames = max_batch_frames
        self.max_batch_size = max_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle_sample = shuffle_sample
        self.shuffle_batch = shuffle_batch
        self.sort_by_similar_size = sort_by_similar_size
        self.size_reversed = size_reversed
        self.frame_count_grid = frame_count_grid
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.batches = None
        self.formed_epoch = None

    def __form_batches(self):
        """Form batches for the current epoch."""
        if self.formed_epoch == self.epoch:
            return

        rng = np.random.default_rng(self.seed + self.epoch)

        # Create indices
        indices = np.arange(len(self.dataset))

        if self.shuffle_sample:
            rng.shuffle(indices)

            if self.sort_by_similar_size:
                # Group by similar sizes for efficient batching
                sizes = np.array(self.dataset.sizes)[indices]
                sizes = (np.round(sizes / self.frame_count_grid) * self.frame_count_grid)
                sizes = sizes.clip(self.frame_count_grid, None)

                if self.size_reversed:
                    sizes *= -1

                indices = indices[np.argsort(sizes, kind='mergesort')]

        indices = indices.tolist()

        # Batch by size
        batches = []
        current_batch = []
        current_frames = 0

        for idx in indices:
            num_frames = self.dataset.num_frames(idx)

            # Check if adding this sample exceeds limits
            if (len(current_batch) > 0 and
                (current_frames + num_frames > self.max_batch_frames or
                 len(current_batch) >= self.max_batch_size)):
                batches.append(current_batch)
                current_batch = []
                current_frames = 0

            current_batch.append(idx)
            current_frames += num_frames

        # Add last batch
        if len(current_batch) > 0 and not self.drop_last:
            batches.append(current_batch)

        # Shuffle batches
        if self.shuffle_batch:
            rng.shuffle(batches)

        # Distribute batches across replicas
        self.batches = batches[self.rank::self.num_replicas]
        self.formed_epoch = self.epoch

    def __iter__(self):
        self.__form_batches()
        for batch in self.batches:
            yield batch

    def __len__(self):
        self.__form_batches()
        return len(self.batches)

    def set_epoch(self, epoch):
        """Set epoch for shuffling."""
        self.epoch = epoch


def build_train_dataloader(dataset, config, shuffle=True):
    """Build the training dataloader described by ``config['training']``.

    Uses :class:`DsBatchSampler` when ``max_batch_frames > 0`` and the dataset
    reports per-item frame counts; otherwise falls back to a fixed batch size.

    Args:
        dataset: Paddle Dataset, ideally with ``sizes`` and ``num_frames()``.
        config: Full training configuration dict.
        shuffle: Shuffle samples (fixed-batch-size path only).

    Returns:
        Paddle DataLoader.
    """
    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 0))
    max_batch_frames = int(train_cfg.get("max_batch_frames", 0))

    if max_batch_frames > 0:
        if hasattr(dataset, "num_frames") and hasattr(dataset, "sizes"):
            batch_sampler = DsBatchSampler(
                dataset,
                max_batch_frames=max_batch_frames,
                max_batch_size=batch_size,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle_sample=shuffle,
                shuffle_batch=shuffle,
                frame_count_grid=int(train_cfg.get("sampler_frame_count_grid", 5000)),
                seed=int(config.get("seed", 0)),
            )
            return build_dataloader(dataset, batch_sampler, num_workers=num_workers)
        print(
            f"  WARNING: max_batch_frames={max_batch_frames} ignored — "
            f"{type(dataset).__name__} exposes no frame counts; "
            f"using batch_size={batch_size}."
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=getattr(dataset, "collater", None),
        num_workers=num_workers,
    )


def build_val_dataloader(dataset, batch_size=1):
    """Build a deterministic validation dataloader.

    Args:
        dataset: Paddle Dataset.
        batch_size: Validation batch size (default 1, unpadded comparison).

    Returns:
        Paddle DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=getattr(dataset, "collater", None),
    )


def build_dataloader(dataset, batch_sampler, num_workers=4, collate_fn=None):
    """Build distributed dataloader with custom batch sampler.

    Args:
        dataset: Paddle Dataset.
        batch_sampler: DsBatchSampler instance.
        num_workers: Number of data loading workers.
        collate_fn: Batch collation function; defaults to ``dataset.collater``
            when the dataset provides one.

    Returns:
        Paddle DataLoader.
    """
    from paddle.io import DataLoader

    if collate_fn is None:
        collate_fn = getattr(dataset, "collater", None)

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        use_shared_memory=True,
    )
