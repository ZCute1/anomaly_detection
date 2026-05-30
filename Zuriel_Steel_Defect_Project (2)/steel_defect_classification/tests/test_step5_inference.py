"""
Step 5 Tests — INFER-1 and INFER-2.

Run with:
    pytest tests/test_step5_inference.py -v

Prerequisites:
    - Step 1 (PREPROCESS-1, PREPROCESS-2) must be completed.
    - Step 3 (MODEL-1, MODEL-2) must be completed.
    These are needed because SteelPredictor internally creates
    val transforms and instantiates SteelCNN.
"""
import numpy as np
import pytest
import torch

from steel_defect.utils import NUM_CLASSES, CLASS_NAMES, IMAGE_SIZE
from steel_defect.model import SteelCNN
from steel_defect.inference import SteelPredictor


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def checkpoint_path(tmp_path):
    """Create a valid checkpoint file from a fresh SteelCNN."""
    model = SteelCNN(num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 1,
        "best_val_acc": 0.5,
        "num_classes": NUM_CLASSES,
    }
    path = tmp_path / "test_checkpoint.pt"
    torch.save(checkpoint, path)
    return path


@pytest.fixture
def predictor(checkpoint_path):
    """Create a SteelPredictor pointing to the test checkpoint."""
    return SteelPredictor(checkpoint_path=checkpoint_path, device=torch.device("cpu"))


@pytest.fixture
def dummy_image():
    """Random RGB image matching expected input format."""
    return np.random.randint(0, 256, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)


# ── INFER-1: load_model ──────────────────────────────────

class TestInfer1LoadModel:
    """INFER-1: SteelPredictor.load_model."""

    def test_model_is_none_before_load(self, checkpoint_path):
        predictor = SteelPredictor(
            checkpoint_path=checkpoint_path, device=torch.device("cpu")
        )
        assert predictor.model is None, "Model should be None before load_model()"

    def test_model_loaded_after_call(self, predictor):
        predictor.load_model()
        assert predictor.model is not None, (
            "self.model should be set after load_model()"
        )

    def test_model_is_steel_cnn(self, predictor):
        predictor.load_model()
        assert isinstance(predictor.model, SteelCNN), (
            f"Expected SteelCNN, got {type(predictor.model).__name__}"
        )

    def test_model_in_eval_mode(self, predictor):
        predictor.load_model()
        assert not predictor.model.training, (
            "Model should be in eval mode after load_model() — call model.eval()"
        )

    def test_missing_checkpoint_raises(self, tmp_path):
        predictor = SteelPredictor(
            checkpoint_path=tmp_path / "nonexistent.pt",
            device=torch.device("cpu"),
        )
        with pytest.raises(FileNotFoundError):
            predictor.load_model()

    def test_model_has_correct_num_classes(self, predictor):
        predictor.load_model()
        # Push a dummy tensor through to check output size
        dummy = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
        with torch.no_grad():
            output = predictor.model(dummy)
        assert output.shape[1] == NUM_CLASSES, (
            f"Model output has {output.shape[1]} classes, expected {NUM_CLASSES}"
        )


# ── INFER-2: predict ─────────────────────────────────────

class TestInfer2Predict:
    """INFER-2: SteelPredictor.predict."""

    def test_returns_dict(self, predictor, dummy_image):
        result = predictor.predict(dummy_image)
        assert isinstance(result, dict), "predict() should return a dict"

    def test_has_required_keys(self, predictor, dummy_image):
        result = predictor.predict(dummy_image)
        required = {"label", "confidence", "class_scores", "predicted_idx", "latency_ms"}
        missing = required - set(result.keys())
        assert not missing, f"Result dict is missing keys: {missing}"

    def test_label_is_valid_class(self, predictor, dummy_image):
        result = predictor.predict(dummy_image)
        assert result["label"] in CLASS_NAMES, (
            f"Label '{result['label']}' is not in CLASS_NAMES"
        )

    def test_confidence_in_range(self, predictor, dummy_image):
        result = predictor.predict(dummy_image)
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"Confidence should be in [0, 1], got {result['confidence']}"
        )

    def test_predicted_idx_is_int(self, predictor, dummy_image):
        result = predictor.predict(dummy_image)
        assert isinstance(result["predicted_idx"], int), (
            f"predicted_idx should be int, got {type(result['predicted_idx'])}"
        )

    def test_predicted_idx_matches_label(self, predictor, dummy_image):
        result = predictor.predict(dummy_image)
        assert CLASS_NAMES[result["predicted_idx"]] == result["label"], (
            "predicted_idx and label should correspond to the same class"
        )

    def test_class_scores_has_all_classes(self, predictor, dummy_image):
        result = predictor.predict(dummy_image)
        for name in CLASS_NAMES:
            assert name in result["class_scores"], (
                f"class_scores missing '{name}'"
            )

    def test_class_scores_sum_to_one(self, predictor, dummy_image):
        result = predictor.predict(dummy_image)
        total = sum(result["class_scores"].values())
        assert abs(total - 1.0) < 0.01, (
            f"class_scores should sum to ~1.0, got {total:.4f} — "
            "did you apply F.softmax?"
        )

    def test_class_scores_are_floats(self, predictor, dummy_image):
        result = predictor.predict(dummy_image)
        for name, score in result["class_scores"].items():
            assert isinstance(score, float), (
                f"Score for '{name}' should be float, got {type(score)}"
            )

    def test_latency_is_positive(self, predictor, dummy_image):
        result = predictor.predict(dummy_image)
        assert result["latency_ms"] > 0, "latency_ms should be positive"

    def test_auto_loads_model(self, checkpoint_path, dummy_image):
        """predict() should auto-call load_model() if model is None."""
        predictor = SteelPredictor(
            checkpoint_path=checkpoint_path, device=torch.device("cpu")
        )
        assert predictor.model is None
        result = predictor.predict(dummy_image)
        assert predictor.model is not None, (
            "predict() should auto-load the model if self.model is None"
        )
        assert "label" in result
