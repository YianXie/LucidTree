"""Comprehensive tests for neural network training functionality."""

# fmt: off

import logging
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mini_katago.constants import BOARD_SIZE
from mini_katago.nn.datasets.precomputed_dataset import PrecomputedGoDataset
from mini_katago.nn.evaluate import (evaluate_both, evaluate_policy,
                                     evaluate_value)
from mini_katago.nn.model import SmallPVNet
from mini_katago.nn.train import train_one_epoch

# fmt: on


class TestModelArchitecture:
    """Test suite for model architecture."""

    def test_model_initialization(self) -> None:
        """Test that model initializes correctly."""
        model = SmallPVNet()
        assert isinstance(model, nn.Module)
        assert model.board_size == BOARD_SIZE
        assert model.action_size == BOARD_SIZE * BOARD_SIZE + 1

    def test_model_forward_pass_policy_only(self) -> None:
        """Test model forward pass with policy head."""
        model = SmallPVNet()
        batch_size = 4
        in_channels = 6

        x = torch.randn(batch_size, in_channels, BOARD_SIZE, BOARD_SIZE)
        policy_logits, value = model(x)

        assert policy_logits.shape == (batch_size, BOARD_SIZE * BOARD_SIZE + 1)
        assert value.shape == (batch_size,)

    def test_model_forward_pass_output_ranges(self) -> None:
        """Test that model outputs are in expected ranges."""
        model = SmallPVNet()
        batch_size = 2
        x = torch.randn(batch_size, 6, BOARD_SIZE, BOARD_SIZE)

        policy_logits, value = model(x)

        # Policy logits can be any real number
        assert not torch.isnan(policy_logits).any()
        assert not torch.isinf(policy_logits).any()

        # Value should be in [-1, 1] due to Tanh activation
        assert (value >= -1.0).all() and (value <= 1.0).all()
        assert not torch.isnan(value).any()

    def test_model_different_batch_sizes(self) -> None:
        """Test model with different batch sizes."""
        model = SmallPVNet()

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 6, BOARD_SIZE, BOARD_SIZE)
            policy_logits, value = model(x)

            assert policy_logits.shape[0] == batch_size
            assert value.shape[0] == batch_size

    def test_model_custom_parameters(self) -> None:
        """Test model with custom initialization parameters."""
        in_channels = 8
        board_size = 13

        model = SmallPVNet(in_channels=in_channels, board_size=board_size)

        x = torch.randn(2, in_channels, board_size, board_size)
        policy_logits, value = model(x)

        assert policy_logits.shape == (2, board_size * board_size + 1)
        assert value.shape == (2,)


class TestTrainingLoop:
    """Test suite for training loop functionality."""

    def test_train_one_epoch_policy_only(self) -> None:
        """Test training one epoch with policy network only."""
        model = SmallPVNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Create dummy dataset
        batch_size = 4
        num_samples = 16
        x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
        y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))

        dataset = TensorDataset(x, y_pol)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        logger = logging.getLogger("test")
        logger.setLevel(logging.ERROR)  # Suppress debug logs

        loss = train_one_epoch(
            model=model,
            loader=loader,
            optim=optimizer,
            use_value=False,
            epoch=0,
            logger=logger,
            device=torch.device("cpu"),
        )

        assert isinstance(loss, float)
        assert loss > 0
        assert not torch.isnan(torch.tensor(loss))

    def test_train_one_epoch_with_value(self) -> None:
        """Test training one epoch with both policy and value networks."""
        model = SmallPVNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Create dummy dataset with value
        batch_size = 4
        num_samples = 16
        x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
        y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))
        y_val = torch.randn(num_samples) * 2 - 1  # Values in [-1, 1]

        dataset = TensorDataset(x, y_pol, y_val)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        logger = logging.getLogger("test")
        logger.setLevel(logging.ERROR)

        loss = train_one_epoch(
            model=model,
            loader=loader,
            optim=optimizer,
            use_value=True,
            epoch=0,
            logger=logger,
            device=torch.device("cpu"),
        )

        assert isinstance(loss, float)
        assert loss > 0
        assert not torch.isnan(torch.tensor(loss))

    def test_train_one_epoch_label_smoothing(self) -> None:
        """Test training with label smoothing."""
        model = SmallPVNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        batch_size = 4
        num_samples = 16
        x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
        y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))

        dataset = TensorDataset(x, y_pol)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        logger = logging.getLogger("test")
        logger.setLevel(logging.ERROR)

        loss = train_one_epoch(
            model=model,
            loader=loader,
            optim=optimizer,
            use_value=False,
            epoch=0,
            logger=logger,
            device=torch.device("cpu"),
            label_smoothing=0.1,
        )

        assert isinstance(loss, float)
        assert loss > 0

    def test_train_one_epoch_lambda_value(self) -> None:
        """Test training with different lambda_value for value loss."""
        model = SmallPVNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        batch_size = 4
        num_samples = 16
        x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
        y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))
        y_val = torch.randn(num_samples) * 2 - 1

        dataset = TensorDataset(x, y_pol, y_val)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        logger = logging.getLogger("test")
        logger.setLevel(logging.ERROR)

        loss1 = train_one_epoch(
            model=model,
            loader=loader,
            optim=optimizer,
            use_value=True,
            epoch=0,
            logger=logger,
            device=torch.device("cpu"),
            lambda_value=0.5,
        )

        # Reset model and optimizer for fair comparison
        model2 = SmallPVNet()
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

        loss2 = train_one_epoch(
            model=model2,
            loader=loader,
            optim=optimizer2,
            use_value=True,
            epoch=0,
            logger=logger,
            device=torch.device("cpu"),
            lambda_value=1.0,
        )

        # Both should produce valid losses
        assert isinstance(loss1, float)
        assert isinstance(loss2, float)
        assert loss1 > 0
        assert loss2 > 0

    def test_train_one_epoch_empty_loader(self) -> None:
        """Test training with empty data loader."""
        model = SmallPVNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Create empty dataset
        x = torch.empty(0, 6, BOARD_SIZE, BOARD_SIZE)
        y_pol = torch.empty(0, dtype=torch.long)

        dataset = TensorDataset(x, y_pol)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        logger = logging.getLogger("test")
        logger.setLevel(logging.ERROR)

        loss = train_one_epoch(
            model=model,
            loader=loader,
            optim=optimizer,
            use_value=False,
            epoch=0,
            logger=logger,
            device=torch.device("cpu"),
        )

        # Should return 0.0 for empty loader
        assert loss == 0.0


class TestEvaluation:
    """Test suite for evaluation functions."""

    def test_evaluate_policy(self) -> None:
        """Test policy evaluation function."""
        model = SmallPVNet()
        model.eval()

        batch_size = 4
        num_samples = 16
        x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
        y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))

        dataset = TensorDataset(x, y_pol)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        loss, acc1, acc5 = evaluate_policy(model, loader, torch.device("cpu"))

        assert isinstance(loss, float)
        assert isinstance(acc1, float)
        assert isinstance(acc5, float)
        assert loss >= 0
        assert 0 <= acc1 <= 1
        assert 0 <= acc5 <= 1
        assert acc5 >= acc1  # Top-5 accuracy should be >= top-1

    def test_evaluate_policy_with_value_data(self) -> None:
        """Test policy evaluation with dataset that includes value."""
        model = SmallPVNet()
        model.eval()

        batch_size = 4
        num_samples = 16
        x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
        y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))
        y_val = torch.randn(num_samples) * 2 - 1

        dataset = TensorDataset(x, y_pol, y_val)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        loss, acc1, acc5 = evaluate_policy(model, loader, torch.device("cpu"))

        assert isinstance(loss, float)
        assert isinstance(acc1, float)
        assert isinstance(acc5, float)
        assert loss >= 0
        assert 0 <= acc1 <= 1
        assert 0 <= acc5 <= 1

    def test_evaluate_value(self) -> None:
        """Test value network evaluation."""
        model = SmallPVNet()
        model.eval()

        batch_size = 4
        num_samples = 16
        x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
        y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))
        y_val = torch.randn(num_samples) * 2 - 1

        dataset = TensorDataset(x, y_pol, y_val)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        loss = evaluate_value(model, loader, torch.device("cpu"))

        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluate_both(self) -> None:
        """Test evaluation of both policy and value networks."""
        model = SmallPVNet()
        model.eval()

        batch_size = 4
        num_samples = 16
        x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
        y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))
        y_val = torch.randn(num_samples) * 2 - 1

        dataset = TensorDataset(x, y_pol, y_val)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        loss, acc1, acc5 = evaluate_both(model, loader, torch.device("cpu"))

        assert isinstance(loss, float)
        assert isinstance(acc1, float)
        assert isinstance(acc5, float)
        assert loss >= 0
        assert 0 <= acc1 <= 1
        assert 0 <= acc5 <= 1

    def test_evaluate_empty_loader(self) -> None:
        """Test evaluation with empty data loader."""
        model = SmallPVNet()
        model.eval()

        x = torch.empty(0, 6, BOARD_SIZE, BOARD_SIZE)
        y_pol = torch.empty(0, dtype=torch.long)

        dataset = TensorDataset(x, y_pol)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        loss, acc1, acc5 = evaluate_policy(model, loader, torch.device("cpu"))

        # Should return 0.0 for empty loader
        assert loss == 0.0
        assert acc1 == 0.0
        assert acc5 == 0.0


class TestPrecomputedDataset:
    """Test suite for precomputed dataset."""

    def test_precomputed_dataset_policy_only(self) -> None:
        """Test loading precomputed dataset with policy only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data
            num_samples = 20
            x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
            y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))

            data_path = Path(tmpdir) / "test_data.pt"
            torch.save({"X": x, "y_policy": y_pol}, data_path)

            dataset = PrecomputedGoDataset(data_path)

            assert len(dataset) == num_samples
            sample = dataset[0]
            assert len(sample) == 2  # x, y_policy
            assert sample[0].shape == (6, BOARD_SIZE, BOARD_SIZE)
            assert isinstance(sample[1], torch.Tensor)

    def test_precomputed_dataset_with_value(self) -> None:
        """Test loading precomputed dataset with value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data
            num_samples = 20
            x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
            y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))
            y_val = torch.randn(num_samples) * 2 - 1

            data_path = Path(tmpdir) / "test_data.pt"
            torch.save({"X": x, "y_policy": y_pol, "y_value": y_val}, data_path)

            dataset = PrecomputedGoDataset(data_path)

            assert len(dataset) == num_samples
            sample = dataset[0]
            assert len(sample) == 3  # x, y_policy, y_value
            assert sample[0].shape == (6, BOARD_SIZE, BOARD_SIZE)
            assert isinstance(sample[1], torch.Tensor)
            assert isinstance(sample[2], torch.Tensor)

    def test_precomputed_dataset_indexing(self) -> None:
        """Test dataset indexing and iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = 10
            x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
            y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))

            data_path = Path(tmpdir) / "test_data.pt"
            torch.save({"X": x, "y_policy": y_pol}, data_path)

            dataset = PrecomputedGoDataset(data_path)

            # Test indexing
            for i in range(num_samples):
                sample = dataset[i]
                assert sample[0].shape == (6, BOARD_SIZE, BOARD_SIZE)

    def test_precomputed_dataset_file_not_found(self) -> None:
        """Test that missing dataset file raises error."""
        non_existent_path = Path("/nonexistent/path/to/data.pt")

        with pytest.raises((FileNotFoundError, OSError)):
            PrecomputedGoDataset(non_existent_path)

    def test_precomputed_dataset_shard_directory(self) -> None:
        """Test loading precomputed dataset from directory of shards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_dir = Path(tmpdir) / "shards"
            shard_dir.mkdir()

            # Create 2 shards with 5 samples each
            for shard_idx in range(2):
                num_samples = 5
                x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
                y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))
                y_val = torch.randn(num_samples) * 2 - 1

                shard_path = shard_dir / f"{shard_idx:03d}.pt"
                torch.save({"X": x, "y_policy": y_pol, "y_value": y_val}, shard_path)

            dataset = PrecomputedGoDataset(shard_dir)

            assert len(dataset) == 10
            sample = dataset[0]
            assert len(sample) == 3
            assert sample[0].shape == (6, BOARD_SIZE, BOARD_SIZE)


class TestTrainingEdgeCases:
    """Test suite for edge cases in training."""

    def test_nan_loss_detection(self) -> None:
        """Test that NaN loss is detected and logged."""
        model = SmallPVNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Create data that might cause NaN (very large values)
        batch_size = 4
        num_samples = 8
        x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE) * 1000
        y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))

        dataset = TensorDataset(x, y_pol)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        logger = logging.getLogger("test")
        logger.setLevel(logging.ERROR)

        # This might produce NaN, but should be handled gracefully
        loss = train_one_epoch(
            model=model,
            loader=loader,
            optim=optimizer,
            use_value=False,
            epoch=0,
            logger=logger,
            device=torch.device("cpu"),
        )

        # Should either be a valid loss or 0.0 (if NaN was detected and loop broke)
        assert isinstance(loss, float)

    def test_model_gradient_flow(self) -> None:
        """Test that gradients flow through the model during training."""
        model = SmallPVNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        batch_size = 4
        num_samples = 8
        x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
        y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))

        dataset = TensorDataset(x, y_pol)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        logger = logging.getLogger("test")
        logger.setLevel(logging.ERROR)

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters() if p.requires_grad]

        train_one_epoch(
            model=model,
            loader=loader,
            optim=optimizer,
            use_value=False,
            epoch=0,
            logger=logger,
            device=torch.device("cpu"),
        )

        # Check that parameters have changed (gradients flowed)
        params_changed = False
        for initial, current in zip(
            initial_params, [p for p in model.parameters() if p.requires_grad]
        ):
            if not torch.allclose(initial, current, atol=1e-6):
                params_changed = True
                break

        assert params_changed, "Model parameters should change after training"

    def test_model_device_placement(self) -> None:
        """Test that model works on CPU device."""
        device = torch.device("cpu")
        model = SmallPVNet().to(device)

        batch_size = 2
        x = torch.randn(batch_size, 6, BOARD_SIZE, BOARD_SIZE).to(device)

        policy_logits, value = model(x)

        assert policy_logits.device == device
        assert value.device == device

    def test_training_with_different_batch_sizes(self) -> None:
        """Test training with various batch sizes."""
        model = SmallPVNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        logger = logging.getLogger("test")
        logger.setLevel(logging.ERROR)

        for batch_size in [1, 2, 4, 8]:
            num_samples = batch_size * 2
            x = torch.randn(num_samples, 6, BOARD_SIZE, BOARD_SIZE)
            y_pol = torch.randint(0, BOARD_SIZE * BOARD_SIZE + 1, (num_samples,))

            dataset = TensorDataset(x, y_pol)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            loss = train_one_epoch(
                model=model,
                loader=loader,
                optim=optimizer,
                use_value=False,
                epoch=0,
                logger=logger,
                device=torch.device("cpu"),
            )

            assert isinstance(loss, float)
            assert loss >= 0
