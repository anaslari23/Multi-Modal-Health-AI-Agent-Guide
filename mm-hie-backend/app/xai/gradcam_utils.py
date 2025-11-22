from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

# Non-interactive backend for server/test environments.
matplotlib.use("Agg")  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam_files(
    rgb_image: np.ndarray,
    grayscale_cam: np.ndarray,
    out_dir: Path,
    base_name: str = "gradcam",
) -> Path:
    """Save raw and blended Grad-CAM visualisations.

    Args:
        rgb_image: HxWx3 float array in [0, 1].
        grayscale_cam: HxW Grad-CAM map (unnormalised or [0, 1]).
        out_dir: Directory to write PNG files into.
        base_name: Base name for output files (without extension).

    Returns:
        Path to the primary blended overlay PNG (for use in analysis/PDF).
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalise CAM to [0, 1].
    cam = np.maximum(grayscale_cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    # Raw heatmap with different colormaps.
    raw_jet_path = out_dir / f"{base_name}_raw_jet.png"
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(cam, cmap="jet")
    plt.tight_layout(pad=0)
    plt.savefig(raw_jet_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    raw_magma_path = out_dir / f"{base_name}_raw_magma.png"
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(cam, cmap="magma")
    plt.tight_layout(pad=0)
    plt.savefig(raw_magma_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Standard blended overlay (jet) using pytorch-grad-cam helper.
    overlay = show_cam_on_image(rgb_image, cam, use_rgb=True)
    overlay_path = out_dir / f"{base_name}_overlay.png"
    from PIL import Image

    Image.fromarray(overlay).save(overlay_path)

    return overlay_path
