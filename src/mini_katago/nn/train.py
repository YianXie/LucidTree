import datetime
import logging
import time
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_katago import utils
from mini_katago.constants import BOARD_SIZE, INFINITY
from mini_katago.nn.datasets.precomputed_dataset import NPZPolicyValueDataset
from mini_katago.nn.evaluate import evaluate
from mini_katago.nn.model import PolicyValueNetwork


def _get_device() -> torch.device:
    """
    Select CUDA if available, else CPU. Used at startup so training uses GPU on EC2 etc.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = _get_device()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optim: torch.optim.Optimizer,
    *,
    epoch: int,
    logger: logging.Logger,
    device: torch.device = device,
    lambda_value: float = 0.5,
    label_smoothing: float = 0.05,
    gradient_accumulation_steps: int = 1,
) -> float:
    """
    Train the model for one epoch

    Args:
        model (nn.Module): the model to train on
        loader (DataLoader[Any]): the DataLoader
        optim (torch.optim.Optimizer): the optimizer
        epoch (int): the current epoch count
        logger (logging.Logger): the logger to log outputs
        device (torch.device, optional): the device type (e.g, cuda or cpu). Defaults to default_device.
        lambda_value (float, optional): the lambda value to be multiplied to the value loss. Defaults to 0.5.
        label_smoothing (float, optional): the label smoothing value for cross_entropy loss. Defaults to 0.05.
        gradient_accumulation_steps (int, optional): accumulate gradients over this many batches before stepping.
            Use > 1 to reduce peak GPU memory (smaller effective batch per step). Defaults to 1.

    Returns:
        float: the average loss
    """
    model.train()
    total = 0.0
    optim.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))  # type: ignore

    logger.info(
        "Epoch %d started. Total batches: %d (grad accum steps: %d).",
        epoch,
        len(loader),
        gradient_accumulation_steps,
    )
    for idx, batch in enumerate(loader):
        x, y_policy, y_value = batch
        x = x.to(device, non_blocking=True)
        y_policy = y_policy.to(device, non_blocking=True)
        y_value = y_value.to(device, non_blocking=True)

        with torch.amp.autocast(  # type: ignore
            device.type, enabled=(device.type == "cuda"), dtype=torch.float16
        ):
            policy_logits, value = model(x)
            policy_loss = F.cross_entropy(
                policy_logits, y_policy, label_smoothing=label_smoothing
            )
            value_loss = F.mse_loss(value, y_value)
            loss = (
                policy_loss + lambda_value * value_loss
            ) / gradient_accumulation_steps

        if torch.isnan(loss):
            logger.error("NaN loss detected at epoch %d", epoch)
            break

        if torch.any(torch.isnan(policy_logits)):
            logger.error("NaN in policy logits at epoch %d", epoch)
            break

        scaler.scale(loss).backward()  # type: ignore

        if (idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        total += float(loss.item()) * gradient_accumulation_steps

        if idx % 1000 == 0:
            logger.info(
                "Epoch %d | Batch %d | loss = %.4f | total_loss = %.4f",
                epoch,
                idx,
                loss.item() * gradient_accumulation_steps,
                total,
            )

    optim.step()
    optim.zero_grad(set_to_none=True)

    return total / max(1, len(loader))


def save_best_model(state: dict[str, Any] | None) -> None:
    """
    Save the best model state

    Args:
        state (dict[str, Any] | None): the best model state
    """
    if state is not None:
        torch.save(state, root / "models/checkpoint_19x19.pt")


if __name__ == "__main__":
    root = utils.get_project_root()
    logger = utils.setup_logger(
        name="training", log_file="training.log", level=logging.INFO
    )

    start_time = time.perf_counter()

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    NUM_EPOCH = 10
    batch_size = 256
    gradient_accumulation_steps = 1

    logger.info("Starting training")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info(
            "CUDA enabled: %s (device %s)",
            torch.cuda.get_device_name(0),
            device,
        )
    else:
        logger.info("CUDA not available; training on CPU")
    logger.info("Total epoch = %d", NUM_EPOCH)
    logger.info("Board size = %d", BOARD_SIZE)
    logger.info("Batch size = %d", batch_size)
    logger.info("Gradient accumulation steps = %d", gradient_accumulation_steps)

    use_cuda = device.type == "cuda"
    pin_memory = use_cuda  # Faster CPU->GPU transfer when using CUDA
    model = PolicyValueNetwork()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.8e-4, weight_decay=2e-4)

    # Run the training and save the losses
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_acc1s: list[float] = []
    val_acc5s: list[float] = []

    best_val_loss = INFINITY
    best_state = None  # Not kept in memory after save; reload from disk when needed
    epoch = 0

    try:
        checkpoint = torch.load(
            root / "models/checkpoint_19x19.pt", map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        epoch = checkpoint["epoch"] + 1
        batch_size = checkpoint["batch_size"]
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        val_acc1s = checkpoint["val_acc1s"]
        val_acc5s = checkpoint["val_acc5s"]
    except FileNotFoundError:
        logger.warning(
            "Checkpoint file does not exist. Starting with no checkpoint file."
        )
    except PermissionError:
        logger.error("Permission denied when accessing the checkpoint file.")

    # Retrieve the datasets
    processed_dir = root / "data/processed"
    train_dataset = NPZPolicyValueDataset(processed_dir / "train/19x19")
    val_dataset = NPZPolicyValueDataset(processed_dir / "val/19x19")
    test_dataset = NPZPolicyValueDataset(processed_dir / "test/19x19")
    logger.info("train_dataset length: %d", len(train_dataset))
    logger.info("val_dataset length: %d", len(val_dataset))
    logger.info("test_dataset length: %d", len(test_dataset))

    # Load the datasets. num_workers=0 avoids extra process memory; pin_memory=False saves host RAM.
    train_loader = DataLoader[Any](
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if use_cuda else 0,
        prefetch_factor=4,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    logger.info("Finished loading train_loader.")

    val_loader = DataLoader[Any](
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if use_cuda else 0,
        prefetch_factor=4,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    logger.info("Finished loading val_loader.")

    test_loader = DataLoader[Any](
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if use_cuda else 0,
        prefetch_factor=4,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    logger.info("Finished loading test_loader.")

    # Training loop
    logger.info("Start training loop.")
    for _ in range(NUM_EPOCH):
        try:
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                epoch=epoch,
                logger=logger,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            train_losses.append(train_loss)

            val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, device=device)
            if use_cuda:
                torch.cuda.empty_cache()
            val_losses.append(val_loss)
            val_acc1s.append(val_acc1)
            val_acc5s.append(val_acc5)

            if val_loss < best_val_loss:
                logger.info("Found a better state at epoch %d", epoch)
                best_val_loss = val_loss
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "batch_size": batch_size,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_acc1s": val_acc1s,
                    "val_acc5s": val_acc5s,
                }
                save_best_model(best_state)
                best_state = None  # Free memory; best checkpoint is on disk

            logger.info(
                "Epoch %d finished | train_loss = %.4f | val_loss = %.4f | val_acc1 = %.4f | val_acc5 = %.4f",
                epoch,
                train_loss,
                val_loss,
                val_acc1,
                val_acc5,
            )
            epoch += 1
        except KeyboardInterrupt:
            logger.info("Training stopped by user at epoch %d", epoch)
            break

    if best_state is not None:
        save_best_model(best_state)

    try:
        checkpoint = torch.load(
            root / "models/checkpoint_19x19.pt", map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        logger.warning("No checkpoint found; running test with last epoch model.")

    test_loss, test_acc1, test_acc5 = evaluate(model, test_loader, device=device)
    logger.info(
        "TEST | loss = %.4f | acc1 = %.4f | acc5 = %.4f",
        test_loss,
        test_acc1,
        test_acc5,
    )

    # Log performance
    end_time = time.perf_counter()
    duration = end_time - start_time
    logger.info(
        "Total training time: %d seconds, or %s",
        duration,
        str(datetime.timedelta(seconds=duration)),
    )

    if epoch > 0:
        # Plot the training overview
        plt.plot(
            range(0, epoch),
            train_losses,
            label="Train Losses",
        )
        plt.plot(
            range(0, epoch),
            val_losses,
            label="Validation Losses",
        )
        plt.plot(
            range(0, epoch),
            val_acc1s,
            label="Validation Accuracy (top 1)",
        )
        plt.plot(
            range(0, epoch),
            val_acc5s,
            label="Validation Accuracy (top 5)",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Losses/Accuracy")

        plt.title("Training Overview")
        plt.legend()

        t = datetime.datetime.now()
        plt.savefig(root / f"figures/{t}.png", dpi=300)

    logger.info("Training end")
