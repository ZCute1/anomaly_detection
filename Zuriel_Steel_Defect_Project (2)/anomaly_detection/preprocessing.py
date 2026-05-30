"""
Image preprocessing pipeline.

Handles all image transformations needed for PatchCore inference:
resizing, normalization, tensor conversion, and augmentation.
"""
import time

import cv2
import numpy as np
import torch
from torchvision import transforms

from anomaly_detection.utils import setup_logging, IMAGE_SIZE

logger = setup_logging(__name__)

# ImageNet normalization (PatchCore uses a pretrained backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transform() -> transforms.Compose:
    """
    Build the standard transform pipeline for inference.

    Returns:
        torchvision Compose transform.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_frame(
    frame: np.ndarray,
    transform: transforms.Compose | None = None,
) -> torch.Tensor:
    """
    Preprocess a raw camera frame for model inference.

    Args:
        frame: RGB numpy array (H, W, 3), uint8.
        transform: Optional custom transform. Uses default if None.

    Returns:
        Preprocessed tensor of shape (1, 3, H, W).
    """
    start = time.perf_counter()

    if transform is None:
        transform = get_inference_transform()

    # Apply transforms
    tensor = transform(frame)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.debug(
        "Preprocessing complete | input=%s | output=%s | time=%.1fms",
        frame.shape,
        tuple(tensor.shape),
        elapsed_ms,
    )

    return tensor


def unnormalize_for_display(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: Normalized tensor (C, H, W) or (1, C, H, W).

    Returns:
        RGB numpy array (H, W, 3), uint8.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    tensor = tensor.cpu() * std + mean
    tensor = torch.clamp(tensor, 0, 1)

    image = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return image


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay an anomaly heatmap on the original image.

    Args:
        image: RGB image (H, W, 3), uint8.
        heatmap: Anomaly score map (H, W), float32 in [0, 1].
        alpha: Blend factor for the heatmap overlay.
        colormap: OpenCV colormap constant.

    Returns:
        Blended RGB image (H, W, 3), uint8.
    """
    # Resize heatmap to match image
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Normalize to 0-255 and apply colormap
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    blended = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    logger.debug("Heatmap overlay applied | alpha=%.2f | size=(%d, %d)", alpha, w, h)
    return blended
