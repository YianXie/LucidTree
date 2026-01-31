import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class PrecomputedGoDataset(Dataset[Any]):
    """
    A class representing a pre-computed dataset
    """

    def __init__(self, path: Path, **kwargs: int) -> None:
        """
        Initialize a pre-computed dataset

        Args:
            path (Path): the path to a .pt file or a directory of shard .pt files
        """
        if path.is_dir():
            shard_paths = sorted(path.glob("*.pt"))
            if not shard_paths:
                raise FileNotFoundError(f"No .pt shards found in {path}")
            xs: list[torch.Tensor] = []
            ys_policy: list[torch.Tensor] = []
            ys_value: list[torch.Tensor] | None = None

            random.shuffle(shard_paths)
            max_shards = kwargs.get("amount", len(shard_paths))
            for shard_path in shard_paths[: min(len(shard_paths), max_shards)]:
                data = torch.load(shard_path, map_location="cpu")
                xs.append(data["X"])
                ys_policy.append(data["y_policy"])
                if "y_value" in data and data["y_value"] is not None:
                    if ys_value is None:
                        ys_value = []
                    ys_value.append(data["y_value"])

            self.X = torch.cat(xs, dim=0)
            self.y_policy = torch.cat(ys_policy, dim=0)
            self.y_value = torch.cat(ys_value, dim=0) if ys_value else None
        else:
            data = torch.load(path, map_location="cpu")
            self.X = data["X"]
            self.y_policy = data["y_policy"]
            self.y_value = data.get("y_value")

    def __len__(self) -> int:
        """
        Get the length of the dataset

        Returns:
            int: the length of the dataset
        """
        return self.X.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """
        Get the data at index

        Args:
            index (int): the index of the data

        Returns:
            tuple[torch.Tensor, ...]: the data
        """
        if self.y_value is None:
            return self.X[index], self.y_policy[index]
        return self.X[index], self.y_policy[index], self.y_value[index]
