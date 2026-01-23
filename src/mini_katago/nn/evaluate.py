from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_policy(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> tuple[float, float]:
    """
    Evaluate the output policy against the validate set

    Args:
        model (nn.Module): the agent model
        loader (DataLoader[Any]): the DataLoader
        device (torch.device): the device type (e.g., "cpu")

    Returns:
        tuple[float, float]: the average loss and accuracy
    """
    model.eval()
    total_loss = 0.0
    correct1 = 0
    total = 0

    for batch in loader:
        x, y = batch  # y: (B,)
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)  # (B, 82)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)  # (B,)
        correct1 += (pred == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / max(1, total)
    acc1 = correct1 / max(1, total)
    return avg_loss, acc1
