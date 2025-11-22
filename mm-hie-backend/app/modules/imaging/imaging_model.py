from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import io
import logging
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from ...schemas import ImagingOutput
from ...xai.gradcam_utils import generate_gradcam_files
from ...utils.s3_client import get_s3_client


logger = logging.getLogger(__name__)


class _EfficientNetHead(nn.Module):
    def __init__(self, in_features: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, embedding_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        emb = self.act(self.fc1(x))
        logits = self.fc2(emb)
        return emb, logits


class ImagingModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

        # Use modern TorchVision weights API. The returned weights object
        # exposes a full preprocessing pipeline via weights.transforms(),
        # which already includes resize, crop, tensor conversion and
        # normalization. This keeps the preprocessing aligned with the
        # pretrained checkpoint and avoids attribute errors such as
        # `ImageClassification` having no `normalize` attribute.
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        embedding_dim = 512

        backbone.classifier = _EfficientNetHead(
            in_features=in_features,
            embedding_dim=embedding_dim,
            num_classes=3,
        )
        self.model = backbone.to(self.device).eval()

        self.embedding_dim = embedding_dim
        self.classes = ["Normal", "Pneumonia", "Edema/Effusion"]

        # Full preprocessing pipeline provided by TorchVision weights.
        # This replaces the previous manual composition that attempted to
        # access `weights.transforms().normalize`, which is not a public
        # attribute on the weights object in newer TorchVision versions.
        self.preprocess = weights.transforms()

        self._target_layer = self.model.features[-1]

    @torch.inference_mode()
    def infer_bytes(self, content: bytes, case_id: Optional[str] = None) -> ImagingOutput:
        """Run imaging inference from raw bytes.

        This method is defensive by design: any error during preprocessing
        or model execution is caught and converted into an empty
        `ImagingOutput` so that the API layer can surface a clean JSON
        error without ever crashing the server process.
        """

        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")

            # Apply the TorchVision-provided preprocessing pipeline and add
            # a batch dimension as required by EfficientNet.
            x = self.preprocess(image).unsqueeze(0).to(self.device)

            emb, logits = self._forward_with_embedding(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)

            probabilities: Dict[str, float] = {
                cls: float(probs[i]) for i, cls in enumerate(self.classes)
            }

            gradcam_path = self._compute_gradcam(image, x, case_id)

            return ImagingOutput(
                probabilities=probabilities,
                gradcam_path=str(gradcam_path) if gradcam_path is not None else None,
                embedding=emb.squeeze(0).tolist(),
            )
        except Exception as exc:  # pragma: no cover - defensive path
            # Log the error for observability but return a safe, empty
            # structure so that upstream FastAPI handlers can translate this
            # into a 5xx JSON response instead of crashing the worker.
            logger.exception("Imaging inference failed: %s", exc)
            return ImagingOutput(probabilities={}, gradcam_path=None, embedding=[])

    def _forward_with_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.model.features(x)
        feats = self.model.avgpool(feats)
        feats = torch.flatten(feats, 1)
        emb, logits = self.model.classifier(feats)
        return emb, logits

    def _compute_gradcam(
        self,
        image: Image.Image,
        x: torch.Tensor,
        case_id: Optional[str],
    ) -> Optional[str]:
        try:
            cam = GradCAM(model=self.model, target_layers=[self._target_layer], use_cuda=False)
            grayscale_cam = cam(input_tensor=x, targets=None)[0]

            rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

            if case_id is not None:
                out_dir = Path("./models/reports") / f"case_{case_id}"
            else:
                out_dir = Path("./models/gradcam")

            # Use the XAI utility to generate raw heatmaps and blended overlay.
            overlay_path = generate_gradcam_files(
                rgb_image=rgb_img,
                grayscale_cam=grayscale_cam,
                out_dir=out_dir,
            )

            # Upload overlay to S3 if available; otherwise fall back to local path.
            try:
                if case_id is not None:
                    key = f"gradcam/{case_id}/{overlay_path.name}"
                else:
                    key = f"gradcam/{overlay_path.name}"
                s3 = get_s3_client()
                s3_key = s3.upload_file(overlay_path, key)
                return s3_key
            except Exception:
                return str(overlay_path)
        except Exception:
            return None
