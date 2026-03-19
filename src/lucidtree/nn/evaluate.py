from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> tuple[float, float, float]:
    """
    Evaluate the output policy against the validate set

    Args:
        model (nn.Module): the agent model
        loader (DataLoader[Any]): the DataLoader
        device (torch.device): the device type (e.g., "cpu")

    Returns:
        tuple[float, float, float]: the average loss, top 1 pick accuracy, and top 5 pick accuracy
    """
    model.eval()
    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0

    for batch in loader:
        x, y_policy, y_value = batch

        non_blocking = device.type == "cuda"
        x = x.to(device, non_blocking=non_blocking)
        y_policy = y_policy.to(device, non_blocking=non_blocking)
        y_value = y_value.to(device, non_blocking=non_blocking)

        policy_logits, value = model(x)
        probs = torch.softmax(policy_logits, dim=1)
        y_value = y_value.view(-1)
        value = value.view(-1)

        policy_loss = F.cross_entropy(policy_logits, y_policy)
        value_loss = F.mse_loss(value, y_value)
        total_loss += (policy_loss.item() + value_loss.item()) * x.size(0)

        top1_prediction = probs.argmax(dim=1)  # (B,)
        _, top5_indices = probs.topk(k=5, dim=1)  # (B, 5)

        correct1 += int((top1_prediction == y_policy).sum().item())
        y_expanded = y_policy.unsqueeze(1)  # (B, 1) for broadcasting with (B, 5)
        correct5 += int(torch.any(top5_indices == y_expanded, dim=1).sum().item())

        total += x.size(0)

    avg_loss = total_loss / max(1, total)
    acc1 = correct1 / max(1, total)
    acc5 = correct5 / max(1, total)
    return avg_loss, acc1, acc5
