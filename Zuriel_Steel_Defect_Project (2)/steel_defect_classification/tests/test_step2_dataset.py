"""
Step 2 Tests — DATA-1, DATA-2, and DATA-3.

Run with:
    pytest tests/test_step2_dataset.py -v

Prerequisites: None (uses dummy images and inline transforms).
"""
import numpy as np
import pytest
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from steel_defect.utils import CLASS_NAMES
from steel_defect.dataset import build_file_list, SteelDataset, create_splits


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def dataset_dir(tmp_path):
    """Create a temporary dataset directory with dummy images."""
    class_sizes = {"no_defect": 10, "defect_1": 6, "defect_2": 4}
    for class_name, count in class_sizes.items():
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(count):
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"img_{i:03d}.png"), img)
    return tmp_path


@pytest.fixture
def file_list(dataset_dir):
    """Build a file list from the dummy dataset."""
    return build_file_list(dataset_dir)


@pytest.fixture
def simple_transform():
    """A minimal transform for testing __getitem__ without depending on Step 1."""
    return A.Compose([
        A.Resize(64, 64),
        ToTensorV2(),
    ])


# ── DATA-1: build_file_list ──────────────────────────────

class TestData1BuildFileList:
    """DATA-1: build_file_list."""

    def test_returns_list(self, dataset_dir):
        result = build_file_list(dataset_dir)
        assert isinstance(result, list)

    def test_correct_total_count(self, dataset_dir):
        result = build_file_list(dataset_dir)
        assert len(result) == 20, f"Expected 20 images (10+6+4), got {len(result)}"

    def test_tuples_have_path_and_label(self, dataset_dir):
        result = build_file_list(dataset_dir)
        for item in result:
            assert isinstance(item, tuple), "Each item should be a tuple"
            assert len(item) == 2, "Each tuple should be (path, label)"
            path, label = item
            assert isinstance(path, str), "Path should be a string"
            assert isinstance(label, int), "Label should be an integer"

    def test_labels_match_class_names(self, dataset_dir):
        result = build_file_list(dataset_dir)
        labels = {label for _, label in result}
        # no_defect=0, defect_1=1, defect_2=2
        assert 0 in labels, "no_defect (label 0) should be present"
        assert 1 in labels, "defect_1 (label 1) should be present"
        assert 2 in labels, "defect_2 (label 2) should be present"

    def test_label_counts(self, dataset_dir):
        result = build_file_list(dataset_dir)
        from collections import Counter
        counts = Counter(label for _, label in result)
        assert counts[0] == 10, "no_defect should have 10 images"
        assert counts[1] == 6, "defect_1 should have 6 images"
        assert counts[2] == 4, "defect_2 should have 4 images"

    def test_ignores_unknown_folders(self, dataset_dir):
        """Folders not in CLASS_NAMES should be skipped."""
        unknown_dir = dataset_dir / "unknown_class"
        unknown_dir.mkdir()
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(unknown_dir / "rogue.png"), img)

        result = build_file_list(dataset_dir)
        assert len(result) == 20, "Unknown folder should be ignored"

    def test_ignores_non_image_files(self, dataset_dir):
        """Non-image files should be skipped."""
        (dataset_dir / "no_defect" / "notes.txt").write_text("not an image")
        result = build_file_list(dataset_dir)
        assert len(result) == 20, ".txt files should be ignored"

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_file_list(tmp_path / "nonexistent")

    def test_paths_point_to_real_files(self, dataset_dir):
        from pathlib import Path
        result = build_file_list(dataset_dir)
        for path, _ in result:
            assert Path(path).exists(), f"Path does not exist: {path}"


# ── DATA-2: SteelDataset.__getitem__ ─────────────────────

class TestData2Getitem:
    """DATA-2: SteelDataset.__getitem__."""

    def test_returns_tuple(self, file_list):
        ds = SteelDataset(file_list, transform=None)
        result = ds[0]
        assert isinstance(result, tuple), "__getitem__ should return a tuple"
        assert len(result) == 2, "Should return (image, label)"

    def test_label_is_int(self, file_list):
        ds = SteelDataset(file_list, transform=None)
        _, label = ds[0]
        assert isinstance(label, (int, np.integer)), "Label should be an integer"

    def test_image_is_numpy_without_transform(self, file_list):
        ds = SteelDataset(file_list, transform=None)
        image, _ = ds[0]
        assert isinstance(image, np.ndarray), (
            "Without transform, image should be a numpy array"
        )

    def test_image_is_rgb(self, file_list):
        """Image should be RGB (3 channels), not BGR."""
        ds = SteelDataset(file_list, transform=None)
        image, _ = ds[0]
        assert image.ndim == 3, "Image should be 3D (H, W, C)"
        assert image.shape[2] == 3, "Image should have 3 channels"

    def test_image_is_tensor_with_transform(self, file_list, simple_transform):
        ds = SteelDataset(file_list, transform=simple_transform)
        image, _ = ds[0]
        assert isinstance(image, torch.Tensor), (
            "With transform including ToTensorV2, image should be a torch.Tensor"
        )

    def test_tensor_shape_with_transform(self, file_list, simple_transform):
        ds = SteelDataset(file_list, transform=simple_transform)
        image, _ = ds[0]
        assert image.shape == (3, 64, 64), (
            f"Expected (3, 64, 64), got {tuple(image.shape)}"
        )

    def test_all_indices_loadable(self, file_list):
        ds = SteelDataset(file_list, transform=None)
        for i in range(len(ds)):
            image, label = ds[i]
            assert image is not None, f"Image at index {i} is None"

    def test_len_matches_file_list(self, file_list):
        ds = SteelDataset(file_list, transform=None)
        assert len(ds) == len(file_list)


# ── DATA-3: create_splits ────────────────────────────────

class TestData3Splits:
    """DATA-3: create_splits."""

    def test_returns_three_lists(self, file_list):
        train, val, test = create_splits(file_list)
        assert isinstance(train, list)
        assert isinstance(val, list)
        assert isinstance(test, list)

    def test_no_data_lost(self, file_list):
        """All samples should end up in exactly one split."""
        train, val, test = create_splits(file_list)
        assert len(train) + len(val) + len(test) == len(file_list), (
            "Total samples across splits should equal the original file list"
        )

    def test_approximate_ratios(self, file_list):
        """Splits should roughly match 70/15/15."""
        train, val, test = create_splits(file_list)
        total = len(file_list)
        train_ratio = len(train) / total
        val_ratio = len(val) / total
        test_ratio = len(test) / total
        assert 0.55 < train_ratio < 0.85, f"Train ratio {train_ratio:.2f} is off"
        assert 0.05 < val_ratio < 0.30, f"Val ratio {val_ratio:.2f} is off"
        assert 0.05 < test_ratio < 0.30, f"Test ratio {test_ratio:.2f} is off"

    def test_no_overlap(self, file_list):
        """No sample should appear in more than one split."""
        train, val, test = create_splits(file_list)
        train_paths = {p for p, _ in train}
        val_paths = {p for p, _ in val}
        test_paths = {p for p, _ in test}
        assert train_paths.isdisjoint(val_paths), "Train and val overlap"
        assert train_paths.isdisjoint(test_paths), "Train and test overlap"
        assert val_paths.isdisjoint(test_paths), "Val and test overlap"

    def test_reproducible(self, file_list):
        """Same seed should produce the same splits."""
        train1, val1, test1 = create_splits(file_list, seed=42)
        train2, val2, test2 = create_splits(file_list, seed=42)
        assert train1 == train2, "Splits should be reproducible with the same seed"
        assert val1 == val2
        assert test1 == test2

    def test_different_seed_different_splits(self, file_list):
        """Different seeds should produce different splits."""
        train1, _, _ = create_splits(file_list, seed=42)
        train2, _, _ = create_splits(file_list, seed=99)
        assert train1 != train2, "Different seeds should produce different splits"

    def test_all_classes_in_train(self, file_list):
        """Stratification should ensure every class appears in training."""
        train, _, _ = create_splits(file_list)
        train_labels = {label for _, label in train}
        original_labels = {label for _, label in file_list}
        assert train_labels == original_labels, (
            "Every class in the dataset should appear in the training split"
        )
