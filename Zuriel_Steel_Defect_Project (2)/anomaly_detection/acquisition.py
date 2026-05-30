"""
Camera acquisition simulator.

Mimics an industrial camera by loading images from the MVTec dataset
as if they were captured from a production line. Implements the
producer-consumer pattern with a frame buffer.

In production, this module would be replaced by a real camera driver
(e.g., Harvesters with a GenTL producer for GigE Vision cameras or vendor specific API like Sapera, Vimba, JAI, etc..).
"""
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from anomaly_detection.utils import setup_logging, MVTEC_TEST_DIR, MVTEC_TRAIN_GOOD

logger = setup_logging(__name__)


class FrameMetadata:
    """Metadata attached to each acquired frame."""

    def __init__(self, frame_id: int, source_path: str, timestamp: float):
        self.frame_id = frame_id
        self.source_path = source_path
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return f"Frame(id={self.frame_id}, source={Path(self.source_path).name})"


class CameraSimulator:
    """
    Simulates industrial camera acquisition from a directory of images.

    Mirrors key concepts from real acquisition systems:
    - Software trigger (acquire_frame) analogous to hardware trigger
    - Circular buffer (deque with maxlen) to prevent unbounded memory use
    - Frame metadata tracking (ID, source, timestamp)
    - Configurable acquisition delay to simulate exposure time

    Usage:
        cam = CameraSimulator("data/metal_nut/test/bent")
        cam.start_acquisition()
        frame, meta = cam.acquire_frame()
        cam.stop_acquisition()
    """

    def __init__(
        self,
        image_dir: str | Path,
        buffer_size: int = 20,
        exposure_delay: float = 0.0,
    ):
        """
        Args:
            image_dir: Path to directory containing images.
            buffer_size: Maximum frames in the circular buffer.
            exposure_delay: Simulated exposure time in seconds.
        """
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            p for p in self.image_dir.rglob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        )

        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}. "
                f"Download the MVTec AD metal_nut dataset first."
            )

        self.buffer: deque = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.exposure_delay = exposure_delay
        self.frame_count = 0
        self._acquiring = False

        logger.info(
            "CameraSimulator initialized | source=%s | images=%d | buffer_size=%d",
            self.image_dir,
            len(self.image_paths),
            buffer_size,
        )

    def start_acquisition(self) -> None:
        """Begin acquisition (analogous to camera.start_grabbing())."""
        self._acquiring = True
        self.frame_count = 0
        self.buffer.clear()
        logger.info("Acquisition started")

    def stop_acquisition(self) -> None:
        """Stop acquisition and clear buffer."""
        self._acquiring = False
        logger.info(
            "Acquisition stopped | total_frames=%d | buffer_remaining=%d",
            self.frame_count,
            len(self.buffer),
        )

    def acquire_frame(self) -> tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        """
        Simulate a single frame capture (software trigger).

        Returns:
            Tuple of (frame as numpy array, frame metadata), or (None, None)
            if acquisition is not active.
        """
        if not self._acquiring:
            logger.warning("acquire_frame called but acquisition is not active")
            return None, None

        # Simulate exposure time
        if self.exposure_delay > 0:
            time.sleep(self.exposure_delay)

        # Cycle through images (like a conveyor belt looping)
        idx = self.frame_count % len(self.image_paths)
        image_path = self.image_paths[idx]

        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.error("Failed to read image: %s", image_path)
            return None, None

        # Convert BGR → RGB for display consistency
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        metadata = FrameMetadata(
            frame_id=self.frame_count,
            source_path=str(image_path),
            timestamp=time.time(),
        )

        self.buffer.append((frame, metadata))
        self.frame_count += 1

        logger.debug(
            "Frame acquired | id=%d | source=%s | buffer=%d/%d",
            metadata.frame_id,
            image_path.name,
            len(self.buffer),
            self.buffer_size,
        )

        return frame, metadata

    def get_buffer_level(self) -> tuple[int, int]:
        """Return (current_count, max_capacity) of the frame buffer."""
        return len(self.buffer), self.buffer_size

    def get_all_test_categories(self) -> list[str]:
        """List available test categories (e.g., 'bent', 'color', 'good')."""
        if not MVTEC_TEST_DIR.exists():
            return []
        return sorted(
            d.name for d in MVTEC_TEST_DIR.iterdir() if d.is_dir()
        )


def get_default_simulator(category: str = "good") -> CameraSimulator:
    """
    Create a simulator pointed at a specific MVTec test category.

    Args:
        category: One of 'good', 'bent', 'color', 'flip', 'scratch'.

    Returns:
        Configured CameraSimulator instance.
    """
    if category == "good":
        image_dir = MVTEC_TRAIN_GOOD
    else:
        image_dir = MVTEC_TEST_DIR / category

    return CameraSimulator(image_dir=image_dir)
