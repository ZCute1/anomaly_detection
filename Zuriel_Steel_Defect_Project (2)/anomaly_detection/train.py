"""
Model training script for PatchCore anomaly detection.

PatchCore does not train in the traditional sense (no gradient descent).
Instead, it extracts features from "good" images using a pretrained
backbone and builds a memory bank (coreset) for nearest-neighbor
anomaly scoring at inference time.

Usage:
    python -m anomaly_detection.train
    python -m anomaly_detection.train --data_root ./data --category metal_nut
"""
import argparse
import time

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

from anomaly_detection.utils import (
    setup_logging,
    DATA_DIR,
    PATCHCORE_MODEL_DIR,
    IMAGE_SIZE,
    BACKBONE,
    DEVICE,
)

logger = setup_logging(__name__)


def train(
    data_root: str | None = None,
    category: str = "metal_nut",
) -> str:
    """
    Run PatchCore feature extraction on MVTec dataset.

    This builds the coreset memory bank from "good" training images.
    The resulting model checkpoint is saved to models/patchcore/.

    Args:
        data_root: Path to MVTec data root. Defaults to project data/ dir.
        category: MVTec AD category to train on.

    Returns:
        Path to the saved model checkpoint.
    """
    data_root = data_root or str(DATA_DIR)

    logger.info("=" * 60)
    logger.info("PatchCore Training")
    logger.info("=" * 60)
    logger.info("Data root:  %s", data_root)
    logger.info("Category:   %s", category)
    logger.info("Backbone:   %s", BACKBONE)
    logger.info("Image size: %s", IMAGE_SIZE)
    logger.info("Device:     %s", DEVICE)
    logger.info("Output:     %s", PATCHCORE_MODEL_DIR)

    start_time = time.time()

    # ── Data Module ────────────────────────────────────────────
    logger.info("Configuring data module...")
    datamodule = MVTecAD(
        root=data_root,
        category=category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,  # Safe default for all platforms
    )

    # ── Model ──────────────────────────────────────────────────
    logger.info("Configuring PatchCore model...")
    model = Patchcore(
        backbone=BACKBONE,
        layers=["layer2", "layer3"],
        num_neighbors=9,
    )

    # ── Engine ─────────────────────────────────────────────────
    logger.info("Starting feature extraction (this may take a few minutes)...")
    engine = Engine(
        default_root_dir=str(PATCHCORE_MODEL_DIR),
        accelerator=DEVICE,
        devices=1,
        max_epochs=1,
    )

    engine.fit(model=model, datamodule=datamodule)

    elapsed = time.time() - start_time
    logger.info("Feature extraction complete | time=%.1fs", elapsed)

    # ── Test ───────────────────────────────────────────────────
    logger.info("Running evaluation on test set...")
    test_results = engine.test(model=model, datamodule=datamodule)
    if test_results:
        for result in test_results:
            for key, value in result.items():
                logger.info("  %s: %.4f", key, value)

    logger.info("Model saved to %s", PATCHCORE_MODEL_DIR)
    return str(PATCHCORE_MODEL_DIR)


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore on MVTec AD")
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Path to MVTec data root directory",
    )
    parser.add_argument(
        "--category", type=str, default="metal_nut",
        help="MVTec AD category (default: metal_nut)",
    )
    args = parser.parse_args()

    train(data_root=args.data_root, category=args.category)


if __name__ == "__main__":
    main()
