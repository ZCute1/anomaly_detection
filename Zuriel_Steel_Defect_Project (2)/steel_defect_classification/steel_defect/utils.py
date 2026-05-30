"""
Shared utilities: paths, logging, device selection, and project constants.
"""
import logging
import sys
from pathlib import Path

import torch

# ── Project Paths ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "steel_defect"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_PATH = MODELS_DIR / "steel_cnn_best.pt"

# ── Dataset Settings ──────────────────────────────────────────
CLASS_NAMES = ["no_defect", "defect_1", "defect_2", "defect_3", "defect_4"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (256, 256)

# ── Device Selection ──────────────────────────────────────────
def get_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# ── Logging ───────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    name: str = "steel_defect",
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

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
