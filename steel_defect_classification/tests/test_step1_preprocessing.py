"""
Step 1 Tests — PREPROCESS-1 and PREPROCESS-2.

Run with:
    pytest tests/test_step1_preprocessing.py -v

Prerequisites: None (this is the first step).
"""
import numpy as np
import pytest
import torch

from steel_defect.utils import IMAGE_SIZE
from steel_defect.preprocessing import build_train_transforms, build_val_transforms


@pytest.fixture
def dummy_image():
    """Create a random RGB image with arbitrary dimensions."""
    return np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)


class TestPreprocess1TrainTransforms:
    """PREPROCESS-1: build_train_transforms."""

    def test_returns_compose(self):
        import albumentations as A
        transform = build_train_transforms()
        assert isinstance(transform, A.Compose)

    def test_output_is_tensor(self, dummy_image):
        transform = build_train_transforms()
        result = transform(image=dummy_image)
        assert isinstance(result["image"], torch.Tensor), (
            "Transform should produce a torch.Tensor — did you include ToTensorV2()?"
        )

    def test_output_shape(self, dummy_image):
        transform = build_train_transforms()
        tensor = transform(image=dummy_image)["image"]
        assert tensor.shape == (3, IMAGE_SIZE[0], IMAGE_SIZE[1]), (
            f"Expected shape (3, {IMAGE_SIZE[0]}, {IMAGE_SIZE[1]}), "
            f"got {tuple(tensor.shape)} — check Resize dimensions"
        )

    def test_output_dtype(self, dummy_image):
        transform = build_train_transforms()
        tensor = transform(image=dummy_image)["image"]
        assert tensor.dtype == torch.float32

    def test_output_is_normalized(self, dummy_image):
        """After ImageNet normalization, values should be roughly in [-3, 3]."""
        transform = build_train_transforms()
        tensor = transform(image=dummy_image)["image"]
        assert tensor.max().item() < 10.0, "Values too large — is Normalize applied?"
        assert tensor.min().item() > -10.0, "Values too small — is Normalize applied?"


class TestPreprocess2ValTransforms:
    """PREPROCESS-2: build_val_transforms."""

    def test_returns_compose(self):
        import albumentations as A
        transform = build_val_transforms()
        assert isinstance(transform, A.Compose)

    def test_output_is_tensor(self, dummy_image):
        transform = build_val_transforms()
        result = transform(image=dummy_image)
        assert isinstance(result["image"], torch.Tensor), (
            "Transform should produce a torch.Tensor — did you include ToTensorV2()?"
        )

    def test_output_shape(self, dummy_image):
        transform = build_val_transforms()
        tensor = transform(image=dummy_image)["image"]
        assert tensor.shape == (3, IMAGE_SIZE[0], IMAGE_SIZE[1])

    def test_output_dtype(self, dummy_image):
        transform = build_val_transforms()
        tensor = transform(image=dummy_image)["image"]
        assert tensor.dtype == torch.float32

    def test_deterministic(self, dummy_image):
        """Validation transforms must produce identical output each time."""
        transform = build_val_transforms()
        result1 = transform(image=dummy_image)["image"]
        result2 = transform(image=dummy_image)["image"]
        assert torch.allclose(result1, result2), (
            "Val transforms should be deterministic — no random augmentation"
        )
