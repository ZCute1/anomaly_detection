"""
Step 4 Tests — TRAIN-1, TRAIN-2, and TRAIN-3.

Run with:
    pytest tests/test_step4_training.py -v

Prerequisites: None (uses a tiny inline model, does not depend on Step 3).

Note: TRAIN-4 (checkpoint saving) is inline code inside train() and cannot
be unit-tested directly. Verify it by running:
    python -m steel_defect.train --epochs 2
and checking that models/steel_cnn_best.pt is created.
"""
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from steel_defect.train import setup_training, train_one_epoch, validate


# ── Fixtures ──────────────────────────────────────────────

NUM_CLASSES = 5
DEVICE = torch.device("cpu")


class TinyModel(nn.Module):
    """Minimal model for testing training functions without depending on SteelCNN."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@pytest.fixture
def tiny_model():
    return TinyModel().to(DEVICE)


@pytest.fixture
def synthetic_loader():
    """DataLoader with random images and labels."""
    images = torch.randn(32, 3, 16, 16)
    labels = torch.randint(0, NUM_CLASSES, (32,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=True)


# ── TRAIN-1: setup_training ──────────────────────────────

class TestTrain1Setup:
    """TRAIN-1: setup_training."""

    def test_returns_tuple(self, tiny_model):
        result = setup_training(tiny_model, learning_rate=1e-3)
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, "Should return (criterion, optimizer)"

    def test_criterion_is_cross_entropy(self, tiny_model):
        criterion, _ = setup_training(tiny_model)
        assert isinstance(criterion, nn.CrossEntropyLoss), (
            f"Expected CrossEntropyLoss, got {type(criterion).__name__}"
        )

    def test_optimizer_is_adam(self, tiny_model):
        _, optimizer = setup_training(tiny_model)
        assert isinstance(optimizer, optim.Adam), (
            f"Expected Adam optimizer, got {type(optimizer).__name__}"
        )

    def test_optimizer_has_model_params(self, tiny_model):
        _, optimizer = setup_training(tiny_model)
        opt_params = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                opt_params.add(id(p))
        model_params = {id(p) for p in tiny_model.parameters()}
        assert model_params == opt_params, (
            "Optimizer should be tracking the model's parameters"
        )

    def test_learning_rate(self, tiny_model):
        _, optimizer = setup_training(tiny_model, learning_rate=0.005)
        lr = optimizer.param_groups[0]["lr"]
        assert lr == 0.005, f"Expected lr=0.005, got {lr}"


# ── TRAIN-2: train_one_epoch ─────────────────────────────

class TestTrain2TrainOneEpoch:
    """TRAIN-2: train_one_epoch."""

    def test_returns_tuple(self, tiny_model, synthetic_loader):
        criterion, optimizer = setup_training(tiny_model)
        result = train_one_epoch(
            tiny_model, synthetic_loader, criterion, optimizer, DEVICE
        )
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, "Should return (loss, accuracy)"

    def test_loss_is_float(self, tiny_model, synthetic_loader):
        criterion, optimizer = setup_training(tiny_model)
        loss, _ = train_one_epoch(
            tiny_model, synthetic_loader, criterion, optimizer, DEVICE
        )
        assert isinstance(loss, float), f"Loss should be float, got {type(loss)}"

    def test_accuracy_is_float(self, tiny_model, synthetic_loader):
        criterion, optimizer = setup_training(tiny_model)
        _, accuracy = train_one_epoch(
            tiny_model, synthetic_loader, criterion, optimizer, DEVICE
        )
        assert isinstance(accuracy, float), (
            f"Accuracy should be float, got {type(accuracy)}"
        )

    def test_accuracy_in_range(self, tiny_model, synthetic_loader):
        criterion, optimizer = setup_training(tiny_model)
        _, accuracy = train_one_epoch(
            tiny_model, synthetic_loader, criterion, optimizer, DEVICE
        )
        assert 0.0 <= accuracy <= 1.0, (
            f"Accuracy should be between 0 and 1, got {accuracy}"
        )

    def test_loss_is_positive(self, tiny_model, synthetic_loader):
        criterion, optimizer = setup_training(tiny_model)
        loss, _ = train_one_epoch(
            tiny_model, synthetic_loader, criterion, optimizer, DEVICE
        )
        assert loss > 0, f"Loss should be positive, got {loss}"

    def test_model_weights_change(self, tiny_model, synthetic_loader):
        """Training should modify model weights."""
        criterion, optimizer = setup_training(tiny_model)
        params_before = [p.clone() for p in tiny_model.parameters()]

        train_one_epoch(tiny_model, synthetic_loader, criterion, optimizer, DEVICE)

        changed = False
        for before, after in zip(params_before, tiny_model.parameters()):
            if not torch.allclose(before, after.data):
                changed = True
                break
        assert changed, "Model weights should change after a training epoch"

    def test_model_in_train_mode_after(self, tiny_model, synthetic_loader):
        """After training, the model should be in train mode."""
        criterion, optimizer = setup_training(tiny_model)
        train_one_epoch(tiny_model, synthetic_loader, criterion, optimizer, DEVICE)
        assert tiny_model.training, (
            "Model should be in train mode after train_one_epoch"
        )


# ── TRAIN-3: validate ────────────────────────────────────

class TestTrain3Validate:
    """TRAIN-3: validate."""

    def test_returns_tuple(self, tiny_model, synthetic_loader):
        criterion, _ = setup_training(tiny_model)
        result = validate(tiny_model, synthetic_loader, criterion, DEVICE)
        assert isinstance(result, tuple)
        assert len(result) == 2, "Should return (loss, accuracy)"

    def test_loss_is_float(self, tiny_model, synthetic_loader):
        criterion, _ = setup_training(tiny_model)
        loss, _ = validate(tiny_model, synthetic_loader, criterion, DEVICE)
        assert isinstance(loss, float)

    def test_accuracy_in_range(self, tiny_model, synthetic_loader):
        criterion, _ = setup_training(tiny_model)
        _, accuracy = validate(tiny_model, synthetic_loader, criterion, DEVICE)
        assert 0.0 <= accuracy <= 1.0

    def test_model_weights_unchanged(self, tiny_model, synthetic_loader):
        """Validation must NOT modify model weights."""
        criterion, _ = setup_training(tiny_model)
        params_before = [p.clone() for p in tiny_model.parameters()]

        validate(tiny_model, synthetic_loader, criterion, DEVICE)

        for before, after in zip(params_before, tiny_model.parameters()):
            assert torch.allclose(before, after.data), (
                "Model weights changed during validation — "
                "make sure you're not calling backward() or optimizer.step()"
            )

    def test_deterministic(self, tiny_model, synthetic_loader):
        """Calling validate twice should produce the same results."""
        criterion, _ = setup_training(tiny_model)
        loss1, acc1 = validate(tiny_model, synthetic_loader, criterion, DEVICE)
        loss2, acc2 = validate(tiny_model, synthetic_loader, criterion, DEVICE)
        assert loss1 == pytest.approx(loss2, abs=1e-5), (
            "Validation should be deterministic"
        )
        assert acc1 == acc2
