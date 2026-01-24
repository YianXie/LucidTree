import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_katago.constants import INFINITY
from mini_katago.go.game import Game
from mini_katago.misc.sgf_parser import parse_sgf_file
from mini_katago.nn.dataset import SgfPolicyValueDataset
from mini_katago.nn.evaluate import evaluate_policy
from mini_katago.nn.model import SmallPVNet
from mini_katago.nn.split import split_game

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optim: torch.optim.Optimizer,
    use_value: bool,
    device: torch.device = device,
    lambda_value: float = 0.5,
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
            loss_pol = F.cross_entropy(policy_logits, y_pol)
            loss_val = F.mse_loss(value_pred, y_val)
            loss = loss_pol + lambda_value * loss_val
        else:
            policy_logits, _ = out
            loss = F.cross_entropy(policy_logits, y_pol)

        loss.backward()  # type: ignore
        optim.step()

        total += float(loss.item())

    return total / max(1, len(loader))


if __name__ == "__main__":
    print("Training start!")

    start_time = time.perf_counter()

    use_value = False
    batch_size = 128
    num_epoch = 100

    games: list[Game] = []
    path = Path("./src/mini_katago/data/")
    for sgf_file in path.iterdir():
        try:
            game = parse_sgf_file(sgf_file)
            games.append(game)
        except ValueError as e:
            print(f"Value error: {e}")
        except Exception as e:
            print(f"Skipped game. Error: {e}")

    train_games, val_games, test_games = split_game(games)
    train_dataset = SgfPolicyValueDataset(train_games, use_value=use_value)
    val_dataset = SgfPolicyValueDataset(val_games, use_value=use_value)
    test_dataset = SgfPolicyValueDataset(test_games, use_value=use_value)

    print("train_dataset length:", len(train_dataset))
    print("val_dataset length:", len(val_dataset))
    print("test_dataset length:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SmallPVNet(seed=0)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # learning rate = 0.001

    # Run the training and save the losses
    losses: list[float] = []
    best_val_loss = INFINITY
    best_state = None
    for epoch in range(num_epoch):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, use_value=use_value, device=device
        )
        losses.append(train_loss)

        val_loss, val_acc1 = evaluate_policy(model, val_loader, device=device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} | val_acc1={val_acc1:.4f}"
            )

    # We found a better state
    if best_state is not None:
        torch.save(best_state, "checkpoint.pt")

    test_loss, test_acc1 = evaluate_policy(model, test_loader, device=device)
    print(f"TEST | loss={test_loss:.4f} | acc1={test_acc1:.4f}")

    end_time = time.perf_counter()
    print(f"Total time spent: {(end_time - start_time):.4f} seconds")

    plt.plot(range(num_epoch), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Error")
    plt.show()

    print("Training end!")
