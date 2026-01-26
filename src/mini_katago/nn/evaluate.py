from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_policy(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> tuple[float, float, float]:
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
    correct5 = 0
    total = 0

    for batch in loader:
        if len(batch) == 2:
            x, y = batch  # y: (B,)
        else:
            x, y, _ = batch  # y: (B,)

        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)  # (B, 82)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        top1_prediction = logits.argmax(dim=1)  # (B,)
        _, top5_indices = logits.topk(k=5, dim=1)  # (B, 5)
        correct1 += (top1_prediction == y).sum().item()
        y_expanded = y.unsqueeze(1)  # (B, 1) for broadcasting with (B, 5)
        correct5 += int(torch.any(top5_indices == y_expanded, dim=1).sum().item())
        total += x.size(0)

    avg_loss = total_loss / max(1, total)
    acc1 = correct1 / max(1, total)
    acc5 = correct5 / max(1, total)
    return avg_loss, acc1, acc5


@torch.no_grad()
def evaluate_value(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> float:
    """
    Evaluate the model's value network

    Args:
        model (nn.Module): the model to evaluate
        loader (DataLoader[Any]): the DataLoader
        device (torch.device): the PyTorch device

    Returns:
        float: the average loss
    """
    model.eval()
    total_loss = 0.0
    total = 0

    for batch in loader:
        x, _, y_value = batch

        x = x.to(device)
        y_value = y_value.to(device).float()

        _, value = model(x)

        y_value = y_value.view(-1)
        value = value.view(-1)

        loss = F.mse_loss(value, y_value)

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    return total_loss / max(1, total)


@torch.no_grad()
def evaluate_both(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Evaluate the model based with both the policy and value network

    Args:
        model (nn.Module): the model to evaluate
        loader (DataLoader[Any]): the DataLoader
        device (torch.device): the PyTorch device

    Returns:
        tuple[float, float, float]: the result of the evaluation
    """
    policy_loss = evaluate_policy(model=model, loader=loader, device=device)
    value_loss = evaluate_value(model=model, loader=loader, device=device)

    return (policy_loss[0] + value_loss, policy_loss[1], policy_loss[2])
