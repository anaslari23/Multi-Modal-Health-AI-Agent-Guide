from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image


def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0, 1)
    return (img * 255).astype("uint8")


def compute_gradcam(model: torch.nn.Module, images: torch.Tensor, class_idx: int, device: torch.device) -> np.ndarray:
    """Very lightweight Grad-CAM implementation for EfficientNet-like models.

    This hooks the last convolutional feature map and its gradients, then
    computes a CAM for the given class index.
    """

    model.eval()
    images = images.to(device)

    activations = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    # assume model.features is the last conv stack (true for EfficientNet)
    handle_fwd = model.features.register_forward_hook(forward_hook)
    handle_bwd = model.features.register_full_backward_hook(backward_hook)  # type: ignore[attr-defined]

    logits = model(images)
    score = logits[:, class_idx].sum()
    model.zero_grad()
    score.backward(retain_graph=True)

    handle_fwd.remove()
    handle_bwd.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Failed to capture activations/gradients for Grad-CAM")

    # Global average pool gradients to get weights
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    cam = torch.relu(cam)

    # Normalize per sample
    cams = []
    for c in cam:
        c_np = c.squeeze().detach().cpu().numpy()
        if c_np.max() > 0:
            c_np = (c_np - c_np.min()) / (c_np.max() - c_np.min())
        else:
            c_np = np.zeros_like(c_np)
        cams.append(c_np)
    return np.stack(cams, axis=0)  # [B, H, W]


def save_gradcam_overlays(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    out_dir: str,
    device: torch.device,
    top_k: int = 1,
) -> None:
    """Save Grad-CAM overlays for a batch of images.

    Saves PNG files under out_dir/case_<idx>.png.
    """

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits = model(images)
        probs = torch.sigmoid(logits)

    for i in range(images.size(0)):
        img = images[i].detach().cpu()
        # choose class index: either true label or top predicted
        if labels.ndim == 1:
            class_idx = int(labels[i].item())
        else:
            true_indices = (labels[i] > 0.5).nonzero(as_tuple=False)
            if true_indices.numel() > 0:
                class_idx = int(true_indices[0].item())
            else:
                class_idx = int(torch.argmax(probs[i]).item())

        cam = compute_gradcam(model, images[i : i + 1], class_idx, device=device)[0]

        # Convert image to uint8 RGB
        img_np = img.cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC
        img_np = _to_uint8((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8))

        # Resize CAM to image resolution before applying color map
        h, w, _ = img_np.shape
        cam_resized = cv2.resize(cam, (w, h))

        heatmap = cv2.applyColorMap(_to_uint8(cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = (0.5 * img_np + 0.5 * heatmap).astype("uint8")

        out_path = Path(out_dir) / f"case_{i}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
