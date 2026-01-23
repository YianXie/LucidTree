import torch
import torch.nn as nn
import torch.nn.functional as F

default_device = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(
    model: nn.Module,
    loader: torch.Tensor,
    optim: torch.optim.Optimizer,
    value_head: bool,
    device: str = default_device,
    lambda_value: float = 0.5,
) -> float:
    """
    Train the model for one epoch

    Args:
        model (nn.Module): the model to train on
        loader (torch.Tensor): the loader (sample data)
        optim (torch.optim.Optimizer): the optimizer
        value_head (bool): the value head
        device (str, optional): the device type (e.g, cuda or cpu). Defaults to default_device.
        lambda_value (float, optional): the lambda value to be multiplied to the value loss. Defaults to 0.5.

    Returns:
        float: the average loss
    """
    model.train()
    total = 0.0
    for batch in loader:
        optim.zero_grad()

        if value_head:
            x, y_pol, y_val = batch
            y_val = y_val.to(device)
        else:
            x, y_pol = batch

        x = x.to(device)
        y_pol = y_pol.to(device)

        out = model(x)
        if value_head:
            policy_logits, value_pred = out
            loss_pol = F.cross_entropy(policy_logits, y_pol)
            loss_val = F.mse_loss(value_pred, y_val)
            loss = loss_pol + lambda_value * loss_val
        else:
            policy_logits = out
            loss = F.cross_entropy(policy_logits, y_pol)

        loss.backward()  # type: ignore
        optim.step()

        total += float(loss.item())

    return total / max(1, len(loader))
