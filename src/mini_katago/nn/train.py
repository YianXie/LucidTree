import logging
import time
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_katago import utils
from mini_katago.constants import BOARD_SIZE, INFINITY, USE_VALUE
from mini_katago.nn.datasets.precomputed_dataset import PrecomputedGoDataset
from mini_katago.nn.evaluate import evaluate_both, evaluate_policy
from mini_katago.nn.model import SmallPVNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optim: torch.optim.Optimizer,
    use_value: bool,
    *,
    epoch: int,
    logger: logging.Logger,
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
        epoch (int): the current epoch count
        device (str, optional): the device type (e.g, cuda or cpu). Defaults to default_device.
        lambda_value (float, optional): the lambda value to be multiplied to the value loss. Defaults to 0.5.
        label_smoothing (float, optional): the label smoothing value for cross_entropy loss. Defaults to 0.05.

    Returns:
        float: the average loss
    """
    model.train()
    total = 0.0
    for batch_idx, batch in enumerate(loader):
        optim.zero_grad()

        if use_value:
            x, y_pol, y_val = batch
            y_val = y_val.to(device)
        else:
            x, y_pol, *_ = batch

        x = x.to(device)
        y_pol = y_pol.to(device)

        policy_logits, value_pred = model(x)
        if use_value:
            loss_pol = F.cross_entropy(
                policy_logits, y_pol, label_smoothing=label_smoothing
            )
            loss_val = F.mse_loss(value_pred, y_val)
            loss = loss_pol + lambda_value * loss_val
        else:
            loss = F.cross_entropy(
                policy_logits, y_pol, label_smoothing=label_smoothing
            )

        if torch.isnan(loss):
            logger.error("NaN loss detected at epoch %d", epoch)
            break

        if torch.any(torch.isnan(policy_logits)):
            logger.error("NaN in policy logits at epoch %d", epoch)

        loss.backward()  # type: ignore
        optim.step()

        total += float(loss.item())

        if batch_idx % 100 == 0:
            logger.debug(
                "Epoch %d | Batch %d | loss = %.4f | total_loss = %.4f",
                epoch,
                batch_idx,
                loss,
                total,
            )

    return total / max(1, len(loader))


def save_best_model(state: dict[str, Any] | None) -> None:
    if state is not None:
        torch.save(state, root / "models/checkpoint.pt")


if __name__ == "__main__":
    root = utils.get_project_root()
    logger = utils.setup_logger(
        name="training", log_file=root / "logs/training.log", level=logging.INFO
    )

    start_time = time.perf_counter()

    torch.manual_seed(0)
    batch_size = 128
    epochs = 30

    logger.info("Starting training")
    logger.info("Board size = %d", BOARD_SIZE)
    logger.info("Batch size = %d", batch_size)
    logger.info("Total epoch = %d", epochs)
    logger.info("USE_VALUE = %s", USE_VALUE)

    processed_dir = root / "data/processed"
    train_dataset = PrecomputedGoDataset(processed_dir / "train", amount=10)
    val_dataset = PrecomputedGoDataset(processed_dir / "val", amount=5)
    test_dataset = PrecomputedGoDataset(processed_dir / "test", amount=5)

    logger.info("train_dataset length: %d", len(train_dataset))
    logger.info("val_dataset length: %d", len(val_dataset))
    logger.info("test_dataset length: %d", len(test_dataset))

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
        best_val_loss = checkpoint["best_val_loss"]
        starting_epoch = (
            checkpoint["epoch"] + 1
        )  # add 1 so we move on from previous epoch
        batch_size = checkpoint["batch_size"]
    except FileNotFoundError:
        logger.warning(
            "Checkpoint file does not exist. Starting with no checkpoint file."
        )
    except PermissionError:
        logger.error("Permission denied when accessing the checkpoint file.")

    for epoch in range(starting_epoch, starting_epoch + epochs):
        try:
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                use_value=USE_VALUE,
                device=device,
                epoch=epoch,
                logger=logger,
            )
            train_losses.append(train_loss)

            if USE_VALUE:
                val_loss, val_acc1, val_acc5 = evaluate_policy(
                    model, val_loader, device=device
                )
            else:
                val_loss, val_acc1, val_acc5 = evaluate_both(
                    model, val_loader, device=device
                )
            val_losses.append(val_loss)
            val_acc1s.append(val_acc1)
            val_acc5s.append(val_acc5)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "batch_size": batch_size,
                    "best_val_loss": best_val_loss,
                }

            if epoch % 5 == 0:
                save_best_model(
                    best_state
                )  # save our best model so we don't lose our progress
                logger.info(
                    "Epoch %d finished | train_loss = %.4f | val_loss = %.4f | val_acc1 = %.4f | val_acc5 = %.4f",
                    epoch,
                    train_loss,
                    val_loss,
                    val_acc1,
                    val_acc5,
                )

        except KeyboardInterrupt:
            logger.info("Training stopped by user at epoch %d", epoch)
            save_best_model(best_state)

    save_best_model(best_state)
    if best_state is not None:
        # Load it and use it for testing
        checkpoint = torch.load(root / "models/checkpoint.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc1, test_acc5 = evaluate_policy(model, test_loader, device=device)
    logger.info(
        "TEST | loss = %.4f | acc1 = %.4f | acc5 = %.4f",
        test_loss,
        test_acc1,
        test_acc5,
    )

    # Log performance
    end_time = time.perf_counter()
    logger.info("Total training time: %.4f seconds", end_time - start_time)

    # Plot the training overview
    plt.plot(
        range(starting_epoch, starting_epoch + epochs),
        train_losses,
        label="Train Losses",
    )
    plt.plot(
        range(starting_epoch, starting_epoch + epochs),
        val_losses,
        label="Validation Losses",
    )
    plt.plot(
        range(starting_epoch, starting_epoch + epochs),
        val_acc1s,
        label="Validation Accuracy (top 1)",
    )
    plt.plot(
        range(starting_epoch, starting_epoch + epochs),
        val_acc5s,
        label="Validation Accuracy (top 5)",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Losses/Accuracy")

    plt.title("Training Overview")
    plt.legend()
    plt.show()

    logger.info("Training end")
