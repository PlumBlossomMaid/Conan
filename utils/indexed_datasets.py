"""IndexedDataset for efficient binary storage.

Adapted from DiffSinger's implementation, using HDF5 for storage.
This provides fast random access for large-scale training datasets.
"""

import pathlib
from collections import deque
from typing import Optional

import h5py
import paddle
import numpy as np


class IndexedDataset:
    """Read-only indexed dataset backed by HDF5.

    Features:
    - Lazy loading: file opened on first access
    - Optional caching: keeps recent items in memory
    - Fast random access: HDF5 optimized for this pattern

    Args:
        path: Directory containing the dataset.
        prefix: Dataset filename prefix (e.g., 'train', 'valid').
        num_cache: Number of items to cache in memory (0 = no cache).
    """

    def __init__(self, path, prefix, num_cache=0):
        super().__init__()
        self.path = pathlib.Path(path) / f'{prefix}.data'
        if not self.path.exists():
            raise FileNotFoundError(f'IndexedDataset not found: {self.path}')
        self.dset = None
        self.cache = deque(maxlen=num_cache)
        self.num_cache = num_cache

    def check_index(self, i):
        if i < 0 or i >= len(self.dset):
            raise IndexError('index out of range')

    def __del__(self):
        if self.dset:
            self.dset.close()

    def __getitem__(self, i):
        """Get item by index.

        Returns:
            Dictionary with numpy arrays/tensors converted from HDF5.
        """
        if self.dset is None:
            self.dset = h5py.File(self.path, 'r')
        self.check_index(i)

        # Check cache
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]

        # Load from HDF5 and convert to Paddle tensors
        item = {
            k: v[()].item() if v.shape == () else paddle.to_tensor(v[()])
            for k, v in self.dset[str(i)].items()
        }

        # Add to cache
        if self.num_cache > 0:
            self.cache.appendleft((i, item))

        return item

    def __len__(self):
        if self.dset is None:
            self.dset = h5py.File(self.path, 'r')
        return len(self.dset)


class IndexedDatasetBuilder:
    """Builder for creating IndexedDataset files.

    Args:
        path: Directory to save the dataset.
        prefix: Dataset filename prefix.
        allowed_attr: Optional whitelist of keys to save (None = save all).
        auto_increment: If True, auto-assign item numbers; if False, require explicit item_no.
    """

    def __init__(self, path, prefix, allowed_attr=None, auto_increment=True):
        self.path = pathlib.Path(path) / f'{prefix}.data'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.dset = h5py.File(self.path, 'w')
        self.counter = 0
        self.auto_increment = auto_increment
        if allowed_attr is not None:
            self.allowed_attr = set(allowed_attr)
        else:
            self.allowed_attr = None

    def add_item(self, item, item_no=None):
        """Add one item to the dataset.

        Args:
            item: Dictionary of arrays/tensors to save.
            item_no: Explicit item number (only if auto_increment=False).

        Returns:
            Item number assigned.
        """
        if self.auto_increment and item_no is not None or not self.auto_increment and item_no is None:
            raise ValueError('auto_increment and provided item_no are mutually exclusive')

        # Filter allowed attributes
        if self.allowed_attr is not None:
            item = {
                k: item[k]
                for k in self.allowed_attr
                if k in item
            }

        # Assign item number
        if self.auto_increment:
            item_no = self.counter
            self.counter += 1

        # Write to HDF5
        for k, v in item.items():
            if v is None:
                continue
            # Convert Paddle tensors to numpy
            if isinstance(v, paddle.Tensor):
                v = v.numpy()
            self.dset.create_dataset(f'{item_no}/{k}', data=v)

        return item_no

    def finalize(self):
        """Close the HDF5 file."""
        self.dset.close()
