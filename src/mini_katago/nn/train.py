import time
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
from torch.utils.data import DataLoader

from mini_katago import utils
from mini_katago.constants import INFINITY, USE_VALUE
from mini_katago.nn.datasets.precomputed_dataset import PrecomputedGoDataset
from mini_katago.nn.evaluate import evaluate_policy
from mini_katago.nn.model import SmallPVNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optim: torch.optim.Optimizer,
    use_value: bool,
    *,
    device: torch.device = device,
    lambda_value: float = 0.5,
    label_smoothing: float = 0.05,
) -> float:
    """
    Train the model for one epoch

    Args:
        model (nn.Module): the model to train on
        loader (torch.Tensor): the DataLoader
        optim (torch.optim.Optimizer): the optimizer
        value_head (bool): the value head
        device (str, optional): the device type (e.g, cuda or cpu). Defaults to default_device.
        lambda_value (float, optional): the lambda value to be multiplied to the value loss. Defaults to 0.5.
        label_smoothing (float, optional): the label smoothing value for cross_entropy loss. Defaults to 0.05.

    Returns:
        float: the average loss
    """
    model.train()
    total = 0.0
    for batch in loader:
        optim.zero_grad()

        if use_value:
            x, y_pol, y_val = batch
            y_val = y_val.to(device)
        else:
            x, y_pol = batch

        x = x.to(device)
        y_pol = y_pol.to(device)

        out = model(x)
        if use_value:
            policy_logits, value_pred = out
            loss_pol = F.cross_entropy(
                policy_logits, y_pol, label_smoothing=label_smoothing
            )
            loss_val = F.mse_loss(value_pred, y_val)
            loss = loss_pol + lambda_value * loss_val
        else:
            policy_logits, _ = out
            loss = F.cross_entropy(
                policy_logits, y_pol, label_smoothing=label_smoothing
            )

        loss.backward()  # type: ignore
        optim.step()

        total += float(loss.item())

    return total / max(1, len(loader))


if __name__ == "__main__":
    print(colored("Training Start!", "green", attrs=["bold"]))

    start_time = time.perf_counter()

    torch.manual_seed(0)
    batch_size = 128
    epochs = 100
    root = utils.get_project_root()

    train_dataset = PrecomputedGoDataset(root / "data/processed/go_9x9_train.pt")
    val_dataset = PrecomputedGoDataset(root / "data/processed/go_9x9_val.pt")
    test_dataset = PrecomputedGoDataset(root / "data/processed/go_9x9_test.pt")

    print("train_dataset length:", len(train_dataset))
    print("val_dataset length:", len(val_dataset))
    print("test_dataset length:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SmallPVNet()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Run the training and save the losses
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_acc1s: list[float] = []
    val_acc5s: list[float] = []

    best_val_loss = INFINITY
    best_state = None
    starting_epoch = 0

    try:
        checkpoint = torch.load(root / "models/checkpoint.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "best_val_loss" in checkpoint:
            best_val_loss = checkpoint["best_val_loss"]
        if "epoch" in checkpoint:
            starting_epoch = checkpoint["epoch"]
    except FileNotFoundError:
        print("Checkpoint file does not exist.")
    except PermissionError:
        print("Permission denied when accessing the checkpoint file.")

    for epoch in range(starting_epoch, starting_epoch + epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, use_value=USE_VALUE, device=device
        )
        train_losses.append(train_loss)

        val_loss, val_acc1, val_acc5 = evaluate_policy(model, val_loader, device=device)
        val_losses.append(val_loss)
        val_acc1s.append(val_acc1)
        val_acc5s.append(val_acc5)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} | val_acc1={val_acc1:.4f} | val_acc5={val_acc5:.4f}"
            )

    # We found a better state
    if best_state is not None:
        # Save the best state
        torch.save(best_state, root / "models/checkpoint.pt")

        # Load it and use it for testing
        checkpoint = torch.load(root / "models/checkpoint.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc1, test_acc5 = evaluate_policy(model, test_loader, device=device)
    print(f"TEST | loss={test_loss:.4f} | acc1={test_acc1:.4f} | acc5={test_acc5:.4f}")

    end_time = time.perf_counter()
    print(f"Total time spent: {(end_time - start_time):.4f} seconds")

    plt.plot(range(epochs), train_losses, label="Train Losses")
    plt.plot(range(epochs), val_losses, label="Validation Losses")
    plt.plot(range(epochs), val_acc1s, label="Validation Accuracy (top 1)")
    plt.plot(range(epochs), val_acc5s, label="Validation Accuracy (top 5)")

    plt.xlabel("Epoch")
    plt.ylabel("Losses/Accuracy")

    plt.title("Training Overview")
    plt.legend()
    plt.show()

    print(colored("Training end!", "green", attrs=["bold"]))
