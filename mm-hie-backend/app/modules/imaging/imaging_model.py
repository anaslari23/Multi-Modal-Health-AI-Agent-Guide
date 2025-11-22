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
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from ...schemas import ImagingOutput
from ...xai.gradcam_utils import generate_gradcam_files
from ...utils.s3_client import get_s3_client


logger = logging.getLogger(__name__)


class ImagingModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

        # Use modern TorchVision weights API.
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.model = resnet50(weights=weights)
        
        # Recreate the custom head from training:
        # Linear(2048, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 2)
        num_features = self.model.fc.in_features  # 2048 for ResNet50
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

        # Load the trained weights
        model_path = Path(__file__).parent / "models" / "xray_classifier_complete.pth"
        try:
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                # The checkpoint contains 'model_state_dict'
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Fallback if it was saved directly
                    self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded X-Ray model from {model_path}")
            else:
                logger.warning(f"Model file not found at {model_path}. Using random weights for head.")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")

        self.model = self.model.to(self.device).eval()

        self.embedding_dim = 512
        self.classes = ["NORMAL", "PNEUMONIA"]

        # Use the preprocessing pipeline from the weights
        self.preprocess = weights.transforms()

        # Target layer for GradCAM (last conv layer of ResNet50)
        self._target_layer = self.model.layer4[-1]

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
            # a batch dimension.
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
        # ResNet50 forward pass up to fc
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Custom head forward pass to extract embedding
        # model.fc is Sequential(Linear, ReLU, Dropout, Linear)
        
        # 1. Linear (2048 -> 512)
        emb = self.model.fc[0](x)
        # 2. ReLU
        emb = self.model.fc[1](emb)
        
        # 3. Dropout
        out = self.model.fc[2](emb)
        # 4. Linear (512 -> 2)
        logits = self.model.fc[3](out)
        
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
