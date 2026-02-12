import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from mini_katago import utils


class NPZPolicyValueDataset(Dataset[Any]):
    """
    A class representing a `.npz` policy-value dataset
    """

    def __init__(self, path: Path, percentage: float = 1.0) -> None:
        """
        Initialize a `.npz` policy-value dataset

        Args:
            path (Path): the path to a directory of shard .npz files
            amount (float | None, optional): the amount of shards to retrieve, defaults to 1.0
        """
        assert percentage > 0.0 and percentage <= 1.0, (
            f"percentage must be within (0, 1], got {percentage}"
        )

        if not path.is_dir():
            raise ValueError("Invalid path. Expecting a directory.")

        shard_paths = sorted(path.glob("*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"No .npz shards found in {path}")

        xs: list[torch.Tensor] = []
        ys_policy: list[torch.Tensor] = []
        ys_value: list[torch.Tensor] = []
        random.shuffle(shard_paths)

        max_shards = int(len(shard_paths) * percentage)
        for shard_path in shard_paths[:max_shards]:
            data = utils.load_npz_dataset(shard_path)
            xs_np = data["X"]
            ys_policy_np = data["y_policy"]
            ys_value_np = data["y_value"]

            xs.append(torch.from_numpy(xs_np).float())
            ys_policy.append(torch.tensor(ys_policy_np, dtype=torch.long))
            ys_value.append(torch.tensor(ys_value_np, dtype=torch.float32))

        self.X = torch.cat(xs, dim=0)
        self.y_policy = torch.cat(ys_policy, dim=0)
        self.y_value = torch.cat(ys_value, dim=0)

    def __len__(self) -> int:
        """
        Get the length of the dataset

        Returns:
            int: the length of the dataset
        """
        return self.X.size(0)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the data at index

        Args:
            index (int): the index of the data

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        return (self.X[index], self.y_policy[index], self.y_value[index])
