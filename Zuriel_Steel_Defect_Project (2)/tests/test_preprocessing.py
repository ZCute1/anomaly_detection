"""Tests for the preprocessing module."""
import numpy as np
import torch

from anomaly_detection.preprocessing import (
    preprocess_frame,
    unnormalize_for_display,
    overlay_heatmap,
    get_inference_transform,
    IMAGE_SIZE,
)


def _make_dummy_frame(h: int = 300, w: int = 300) -> np.ndarray:
    """Create a random RGB uint8 image."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


class TestPreprocessFrame:
    def test_output_shape(self):
        frame = _make_dummy_frame()
        tensor = preprocess_frame(frame)
        assert tensor.shape == (1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])

    def test_output_dtype(self):
        frame = _make_dummy_frame()
        tensor = preprocess_frame(frame)
        assert tensor.dtype == torch.float32

    def test_batch_dimension(self):
        frame = _make_dummy_frame()
        tensor = preprocess_frame(frame)
        assert tensor.dim() == 4
        assert tensor.shape[0] == 1

    def test_different_input_sizes(self):
        for h, w in [(100, 100), (500, 300), (256, 1600)]:
            frame = _make_dummy_frame(h, w)
            tensor = preprocess_frame(frame)
            assert tensor.shape == (1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])


class TestUnnormalize:
    def test_output_range(self):
        frame = _make_dummy_frame()
        tensor = preprocess_frame(frame).squeeze(0)
        restored = unnormalize_for_display(tensor)
        assert restored.dtype == np.uint8
        assert restored.min() >= 0
        assert restored.max() <= 255

    def test_output_shape(self):
        frame = _make_dummy_frame()
        tensor = preprocess_frame(frame).squeeze(0)
        restored = unnormalize_for_display(tensor)
        assert restored.shape == (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)


class TestOverlayHeatmap:
    def test_output_shape(self):
        image = _make_dummy_frame(256, 256)
        heatmap = np.random.rand(256, 256).astype(np.float32)
        result = overlay_heatmap(image, heatmap)
        assert result.shape == image.shape

    def test_different_heatmap_size(self):
        image = _make_dummy_frame(256, 256)
        heatmap = np.random.rand(64, 64).astype(np.float32)
        result = overlay_heatmap(image, heatmap)
        assert result.shape == image.shape

    def test_alpha_zero_returns_original(self):
        image = _make_dummy_frame(256, 256)
        heatmap = np.random.rand(256, 256).astype(np.float32)
        result = overlay_heatmap(image, heatmap, alpha=0.0)
        np.testing.assert_array_equal(result, image)
