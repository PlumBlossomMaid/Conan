"""通用分桶函数 — DiffSinger-style batch_by_size"""
from typing import Callable, List, Sequence, TypeVar

T = TypeVar("T")


def batch_by_files(
    items: Sequence[T],
    num_frames_fn: Callable[[T], int],
    *,
    max_batch_frames: int = 50000,
    max_batch_size: int = 32,
    sort_by_len: bool = True,
    grid: int = 6,
) -> List[List[T]]:
    """Group items into batches under frame-count & size constraints.

    Strategy (from DiffSinger DsBatchSampler):
        1. Sort items by approximate length (grid quantization).
        2. Greedy pack: add items until max_batch_frames or max_batch_size is hit.

    Args:
        items: Full list of items (file paths, indices, etc.).
        num_frames_fn: Returns frame count for an item.
        max_batch_frames: Max total frames in a batch (= longest * batch_size).
        max_batch_size: Max items per batch (hard cap).
        sort_by_len: Sort by length before batching (always True for extraction).
        grid: Quantization granularity (frames). Lower = stricter sort.

    Returns:
        batches: List of batches, each being a sub-list of items.
    """
    if not items:
        return []

    # Compute frame counts
    sizes = [num_frames_fn(item) for item in items]

    if sort_by_len:
        # Grid quantization: group items within `grid` frames as "same length"
        approx = [(max(round(s / grid), 1) * grid) for s in sizes]
        # Sort descending (longest first) → larger items packed first, less tail waste
        order = sorted(range(len(items)), key=lambda i: approx[i], reverse=True)
    else:
        order = list(range(len(items)))

    batches = []
    cur_batch: List[T] = []
    cur_max_len = 0

    def is_full():
        if len(cur_batch) >= max_batch_size:
            return True
        next_len = max(cur_max_len, sizes[order[len(cur_batch)]])
        return (len(cur_batch) + 1) * next_len > max_batch_frames

    for idx in order:
        item_len = sizes[idx]
        if not cur_batch:
            # Start new batch
            cur_batch.append(items[idx])
            cur_max_len = item_len
        elif is_full():
            batches.append(cur_batch)
            cur_batch = [items[idx]]
            cur_max_len = item_len
        else:
            cur_batch.append(items[idx])
            cur_max_len = max(cur_max_len, item_len)

    if cur_batch:
        batches.append(cur_batch)

    return batches
