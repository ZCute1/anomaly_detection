"""Tests for the inference module."""
import pytest
from pathlib import Path

from anomaly_detection.inference import DefectDetector
from anomaly_detection.utils import PATCHCORE_MODEL_DIR


class TestDefectDetector:
    def test_init_default(self):
        detector = DefectDetector()
        assert detector.model_dir == PATCHCORE_MODEL_DIR
        assert detector.threshold == 0.5
        assert detector.model is None

    def test_init_custom_threshold(self):
        detector = DefectDetector(threshold=0.7)
        assert detector.threshold == 0.7

    def test_load_model_without_checkpoint_raises(self, tmp_path):
        detector = DefectDetector(model_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="No checkpoint found"):
            detector.load_model()

    def test_predict_without_load_triggers_auto_load(self, tmp_path):
        """predict() should attempt to load the model automatically."""
        import numpy as np

        detector = DefectDetector(model_dir=tmp_path)
        dummy_image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)

        # Should raise because no checkpoint exists, but it should
        # attempt loading (not fail on a different error)
        with pytest.raises(FileNotFoundError, match="No checkpoint found"):
            detector.predict(dummy_image)
