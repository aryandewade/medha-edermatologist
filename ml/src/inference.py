"""
Inference Engine and Triage Rules for Mobile Spacer Skin Screening
Supports 6-class evaluation: Eczema, Psoriasis, Tinea, Acne, Healthy, Suspicious Lesion
"""

import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch

# Project root (…/Medha); used to locate the shared calibration artifact.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPERATURE_PATH = str(_PROJECT_ROOT / "backend" / "models" / "temperature.json")

from model import build_model, SkinDiseaseEfficientNetB0
from preprocess import preprocess_frame_ex, check_image_quality


CLASS_NAMES = [
    "eczema", 
    "psoriasis", 
    "tinea", 
    "acne", 
    "healthy", 
    "suspicious_lesion"
]

CLASS_LABELS = {
    "eczema": "Eczema (Atopic Derm.)",
    "psoriasis": "Psoriasis",
    "tinea": "Tinea (Ringworm)",
    "acne": "Acne Vulgaris",
    "healthy": "Healthy Skin",
    "suspicious_lesion": "Suspicious Lesion"
}

CLASS_COLORS = {
    "eczema": (235, 99, 37),             # Blue
    "psoriasis": (6, 119, 217),          # Amber
    "tinea": (237, 58, 124),            # Purple
    "acne": (105, 150, 5),              # Teal
    "healthy": (129, 185, 16),          # Emerald Green
    "suspicious_lesion": (38, 38, 220)  # Crimson Red
}


class MobileSpacerInferenceEngine:
    """
    Evaluates smartphone spacer photos through quality verification,
    colour constancy, lesion localization, and calibrated clinical triage.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        uncertainty_threshold: float = 0.50,
        urgent_suspicious_threshold: float = 0.25,
        temperature_path: Optional[str] = None
    ):
        self.device = torch.device(device)
        self.uncertainty_threshold = uncertainty_threshold
        self.urgent_suspicious_threshold = urgent_suspicious_threshold

        print(f"[InferenceEngine] Loading 6-class EfficientNet-B0 on {self.device}...")
        self.model = build_model(
            num_classes=len(CLASS_NAMES),
            pretrained=(checkpoint_path is None),
            checkpoint_path=checkpoint_path,
            device=device
        )
        self.model.eval()

        # Load the post-hoc temperature-scaling calibration (Task 2.3 artifact).
        self.temperature = 1.0
        resolved = temperature_path or os.getenv("TEMPERATURE_PATH") or DEFAULT_TEMPERATURE_PATH
        self._load_temperature(resolved)

    def _load_temperature(self, path: str) -> None:
        """Apply a stored temperature to the model if the artifact exists."""
        try:
            from calibrate import load_temperature
            value = load_temperature(path)
            if value and value > 0:
                self.temperature = float(value)
                self.model.temperature.data = torch.tensor([self.temperature])
                print(f"[InferenceEngine] Loaded calibration temperature T={self.temperature} from {path}")
            else:
                print(f"[InferenceEngine] No calibration artifact at {path}; using T=1.0")
        except Exception as exc:  # pragma: no cover - defensive, optional dependency
            print(f"[InferenceEngine] Temperature calibration unavailable ({exc}); using T=1.0")

    @torch.no_grad()
    def predict_image(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation:
          1. Quality Gate (blur, brightness, glare)
          2. Colour Constancy (Shades-of-Gray)
          3. Lesion Localization & ROI Crop
          4. EfficientNet-B0 Forward Pass
          5. Clinical Triage Decision Rules
        """
        t0 = time.perf_counter()

        # 1. Quality evaluation
        quality = check_image_quality(image_bgr)

        # 2. Preprocess frame (reference-patch colour constancy + ROI localisation)
        tensor, roi_rgb, colour_meta = preprocess_frame_ex(
            image_bgr,
            target_size=(224, 224),
            apply_colour_constancy=True,
            is_bgr=True,
            use_reference_patch=True
        )
        tensor = tensor.to(self.device)

        # Surface real calibration telemetry to the API layer.
        quality["reference_patch_found"] = colour_meta["reference_patch_found"]
        quality["colour_cast"] = colour_meta["colour_cast"]
        quality["patch_metrics"] = colour_meta["patch_metrics"]

        # 3. Model forward pass
        probs_tensor = self.model.predict_proba(tensor)
        probs = probs_tensor.cpu().squeeze(0).numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Format class probabilities & top predictions
        prob_dict = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
        sorted_indices = np.argsort(probs)[::-1]
        top_idx = int(sorted_indices[0])
        top_class = CLASS_NAMES[top_idx]
        top_confidence = float(probs[top_idx])

        top3 = [
            {
                "class_id": CLASS_NAMES[i],
                "label": CLASS_LABELS[CLASS_NAMES[i]],
                "probability": round(float(probs[i]), 4)
            }
            for i in sorted_indices[:3]
        ]

        # 5. Clinical Triage Rules
        suspicious_prob = prob_dict["suspicious_lesion"]
        margin = float(probs[sorted_indices[0]] - probs[sorted_indices[1]])
        
        # Rule 1: High-sensitivity Urgent Referral Gate
        if suspicious_prob >= self.urgent_suspicious_threshold:
            advice_level = "urgent"
            advice_title = "Specialist Referral Recommended"
            advice_msg = "Morphological features suggest a potential suspicious or neoplastic lesion. Prompt in-person dermatologist review is advised."
        
        # Rule 2: Image Quality Defect
        elif not quality["passed"]:
            advice_level = "uncertain"
            advice_title = "Capture Quality Warning"
            advice_msg = "Image quality suboptimal: " + "; ".join(quality["issues"])

        # Rule 3: Clinical Uncertainty / Lookalike Ambiguity
        elif (top_confidence < self.uncertainty_threshold) or (margin < 0.10):
            advice_level = "uncertain"
            advice_title = "Uncertain Presentation"
            advice_msg = "Features are ambiguous or borderline between conditions. Consult a dermatologist for physical examination."

        # Rule 4: Confident Finding
        else:
            advice_level = "ok"
            advice_title = f"Features Consistent with {CLASS_LABELS[top_class]}"
            if top_class == "healthy":
                advice_msg = "Skin presentation appears normal with no active inflammatory lesions detected."
            else:
                advice_msg = f"Presentation aligns with {CLASS_LABELS[top_class]}. Standard primary care management recommended."

        return {
            "prediction": {
                "class_id": top_class,
                "label": CLASS_LABELS[top_class],
                "confidence": round(top_confidence, 4),
                "severity": "urgent" if top_class == "suspicious_lesion" else ("normal" if top_class == "healthy" else "common")
            },
            "top3": top3,
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "quality": quality,
            "advice": {
                "level": advice_level,
                "title": advice_title,
                "message": advice_msg
            },
            "latency_ms": round(latency_ms, 1),
            "roi_preview": roi_rgb
        }
