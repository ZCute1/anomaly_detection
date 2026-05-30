"""
Shared utilities: logging configuration, paths, and project settings.
"""
import logging
import sys
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# MVTec Metal Nut paths
MVTEC_DIR = DATA_DIR / "metal_nut"
MVTEC_TRAIN_GOOD = MVTEC_DIR / "train" / "good"
MVTEC_TEST_DIR = MVTEC_DIR / "test"
MVTEC_GROUND_TRUTH = MVTEC_DIR / "ground_truth"

# Model artifacts
PATCHCORE_MODEL_DIR = MODELS_DIR / "patchcore"
PATCHCORE_WEIGHTS = PATCHCORE_MODEL_DIR / "model.ckpt"

# ── Image Settings ─────────────────────────────────────────────
IMAGE_SIZE = (256, 256)
BACKBONE = "wide_resnet50_2"

# ── Device Selection ───────────────────────────────────────────
import torch

def get_device() -> str:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return "gpu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "gpu"  # Anomalib uses Lightning, which maps MPS via 'gpu'
    return "cpu"

DEVICE = get_device()

# ── Logging ────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    name: str = "anomaly_detection",
    level: int = logging.INFO,
    log_file: str = "app.log",
) -> logging.Logger:
    """
    Configure project-wide logging to console and file.

    Args:
        name: Logger name (use __name__ from calling module).
        level: Logging level.
        log_file: Filename inside the logs/ directory.

    Returns:
        Configured logger instance.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
