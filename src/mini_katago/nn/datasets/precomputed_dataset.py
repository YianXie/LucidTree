from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZPolicyValueDataset(Dataset[Any]):
    """
    A class representing a `.npz` dataset. Uses memory-mapped files so that
    only the requested sample(s) are brought into RAM, avoiding OOM on large shards.
    """

    def __init__(
        self, path: Path, percentage: float = 1.0, shard_cache_size: int = 4
    ) -> None:
        """
        Initialize a dataset from a given directory containing `.npz` files

        Args:
            path (Path): the path to the directory containing all the NPZ datasets
            percentage (float, optional): what percent of NPZ datasets should be loaded. Defaults to 1.0.
            shard_cache_size (int, optional): Max number of open memory-mapped NPZ files to cache.
                Only file handles are cached, not array data, so this can be larger than before. Defaults to 4.

        Raises:
            ValueError: if directory path is invalid
            FileNotFoundError: if no .npz shards are found
        """
        assert 0.0 < percentage <= 1.0
        if not path.is_dir():
            raise ValueError("Invalid path. Expecting a directory.")

        shard_paths = sorted(path.glob("*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"No .npz shards found in {path}")

        random.shuffle(shard_paths)
        max_shards = max(1, int(len(shard_paths) * percentage))
        self.shard_paths = shard_paths[:max_shards]

        # Build global index: for each sample idx -> (shard_id, local_idx)
        self._index: list[tuple[int, int]] = []
        self._shard_lengths: list[int] = []

        for sid, sp in enumerate(self.shard_paths):
            with np.load(sp, mmap_mode="r") as data:
                n = int(data["X"].shape[0])
            self._shard_lengths.append(n)
            self._index.extend((sid, i) for i in range(n))

        # Cache open NpzFile handles (memory-mapped); no array data is loaded into RAM.
        self._cache_size = max(1, shard_cache_size)
        self._cache: OrderedDict[int, Any] = OrderedDict()

    def __len__(self) -> int:
        return len(self._index)

    def _get_shard(self, shard_id: int) -> Any:
        """Return an open memory-mapped NpzFile for the shard (LRU cache)."""
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]

        while len(self._cache) >= self._cache_size:
            _, old_file = self._cache.popitem(last=False)
            old_file.close()

        sp = self.shard_paths[shard_id]
        f = np.load(sp, mmap_mode="r")
        self._cache[shard_id] = f
        return f

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shard_id, local_idx = self._index[index]
        data = self._get_shard(shard_id)

        x = np.asarray(data["X"][local_idx].copy(), dtype=np.float32)
        y_p = int(data["y_policy"][local_idx])
        y_v = float(data["y_value"][local_idx])
        return (
            torch.from_numpy(x),
            torch.tensor(y_p, dtype=torch.int64),
            torch.tensor(y_v, dtype=torch.float32),
        )
