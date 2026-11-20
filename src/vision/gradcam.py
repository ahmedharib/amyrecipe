"""Grad-CAM activation heatmap visualization for ConvNets (Area A)."""

from __future__ import annotations

import base64
import io
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAM:
    """Compute Grad-CAM saliency maps for a target convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        self.activations = out.detach()

    def _save_gradient(self, _module: nn.Module, _grad_in: tuple, grad_out: tuple) -> None:
        self.gradients = grad_out[0].detach()

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> Tuple[np.ndarray, int, float]:
        """Return (heatmap in [0, 1], predicted_class_idx, confidence)."""
        self.model.eval()
        x = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, class_idx].item())

        self.model.zero_grad(set_to_none=True)
        score = logits[0, class_idx]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam, class_idx, confidence


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a [0, 1] Grad-CAM heatmap onto an RGB PIL image."""
    rgb = np.array(original_image.convert("RGB"))
    h, w = rgb.shape[:2]
    cam_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_uint8 = np.uint8(255 * np.clip(cam_resized, 0.0, 1.0))
    colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = np.uint8((1.0 - alpha) * rgb + alpha * colored_rgb)
    return Image.fromarray(blended)


def pil_to_base64_png(image: Image.Image) -> str:
    """Encode PIL Image as a data URL string for JSON responses."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
