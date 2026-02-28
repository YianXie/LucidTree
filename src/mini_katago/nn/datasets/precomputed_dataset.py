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
    A class representing a `.npz` dataset
    """

    def __init__(
        self, path: Path, percentage: float = 1.0, shard_cache_size: int = 2
    ) -> None:
        """
        Initialize a dataset from a given directory containing `.npz` files

        Args:
            path (Path): the path to the directory containing all the NPZ datasets
            percentage (float, optional): what percent of NPZ datasets should be loaded. Defaults to 1.0.
            shard_cache_size (int, optional): Shard cache size. Defaults to 2.

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
            with np.load(sp) as data:
                n = int(data["X"].shape[0])
            self._shard_lengths.append(n)
            self._index.extend((sid, i) for i in range(n))

        # Small cache so repeated access doesn't re-decompress constantly
        self._cache_size = max(1, shard_cache_size)
        self._cache: OrderedDict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = OrderedDict()

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(
        self, shard_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # LRU cache hit
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]

        sp = self.shard_paths[shard_id]
        with np.load(sp) as data:
            # Convert to torch once per shard load
            X = torch.from_numpy(data["X"]).to(torch.float32)
            y_policy = torch.from_numpy(data["y_policy"]).to(torch.int64)
            y_value = torch.from_numpy(data["y_value"]).to(torch.float32)

        # Add to cache
        self._cache[shard_id] = (X, y_policy, y_value)
        self._cache.move_to_end(shard_id)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return X, y_policy, y_value

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shard_id, local_idx = self._index[index]
        X, y_policy, y_value = self._load_shard(shard_id)
        return X[local_idx], y_policy[local_idx], y_value[local_idx]
