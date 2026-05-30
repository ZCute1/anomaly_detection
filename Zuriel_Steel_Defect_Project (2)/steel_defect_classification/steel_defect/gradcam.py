"""
Grad-CAM visualization for CNN models.

Generates class activation maps showing which image regions the model
focused on when making its prediction. Works by hooking into the last
convolutional layer and combining its activations with the gradient
of the predicted class.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization", ICCV 2017.

This file is provided as scaffold — students can study it to understand
hooks, gradients, and visualization but do not need to modify it.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for any CNN with a sequential feature extractor.

    Usage:
        cam = GradCAM(model, target_layer=model.features[-1])
        heatmap = cam.generate(input_tensor)
        blended = cam.overlay(image, heatmap)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: The CNN model (must have a forward() method).
            target_layer: The convolutional layer to hook into
                          (typically the last Conv2d in the feature extractor).
        """
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None

        # Register forward hook to capture activations
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        # Register backward hook to capture gradients
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook: store the layer's output activations."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: store the gradients flowing back through the layer."""
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the input.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, C, H, W).
            target_class: Class index to generate the map for.
                          If None, uses the model's predicted class.

        Returns:
            Heatmap as a (H, W) float32 array in [0, 1], where H and W
            match the input image dimensions (not the feature map).
        """
        self.model.eval()

        # Enable gradients for this pass (even in eval mode)
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero existing gradients and backward pass for the target class
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Global average pool the gradients → (C,) channel importance weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # Only keep positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        # Resize to input image dimensions
        _, _, h, w = input_tensor.shape
        cam = cv2.resize(cam, (w, h))

        return cam.astype(np.float32)

    @staticmethod
    def overlay(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay the Grad-CAM heatmap on the original image.

        Args:
            image: RGB image (H, W, 3), uint8.
            heatmap: Grad-CAM heatmap (H, W), float32 in [0, 1].
            alpha: Blend factor for the overlay.
            colormap: OpenCV colormap to apply.

        Returns:
            Blended RGB image (H, W, 3), uint8.
        """
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return blended

    def remove_hooks(self):
        """Remove the registered hooks (call when done)."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()
