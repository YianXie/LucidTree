from pathlib import Path
from typing import Any
import torch

from torch.utils.data import Dataset


class PrecomputedGoDataset(Dataset[Any]):
    """
    A class representing a pre-computed dataset
    """

    def __init__(self, path: Path) -> None:
        """
        Initialize a pre-computed dataset

        Args:
            path (Path): the path to the .pt file
        """
        data = torch.load(path, map_location="cpu")
        self.X: torch.Tensor = data["X"]
        self.y_policy: torch.Tensor = data["y_policy"]
        self.y_value: torch.Tensor = data.get("y_value")

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
