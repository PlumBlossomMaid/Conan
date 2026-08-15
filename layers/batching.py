"""Shared padded-frame batch grouping for Conan."""
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
    """Group items by padded frame budget."""
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if not items:
        return []

    sizes = [num_frames_fn(item) for item in items]

    if sort_by_len:
        grid = max(int(grid), 1)
        approx = [max(round(size / grid), 1) * grid for size in sizes]
        order = sorted(range(len(items)), key=lambda i: (approx[i], i))
    else:
        order = list(range(len(items)))

    batches = []
    current: List[T] = []
    current_max_len = 0

    for idx in order:
        item_len = sizes[idx]
        next_max_len = max(current_max_len, item_len)
        next_cost = next_max_len * (len(current) + 1)
        exceeds_size = len(current) >= max_batch_size
        exceeds_frames = max_batch_frames > 0 and next_cost > max_batch_frames
        if current and (exceeds_size or exceeds_frames):
            batches.append(current)
            current = []
            current_max_len = 0
            next_max_len = item_len

        current.append(items[idx])
        current_max_len = next_max_len

    if current:
        batches.append(current)

    return batches
