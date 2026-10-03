"""
SkinNet 3-CNN Ensemble Classifier Service
Ensemble of EfficientNet-B0, ResNet-18, and MobileNetV2 for 8 acute infectious skin diseases:
- Cellulitis
- Impetigo
- Athlete's Foot (Tinea Pedis)
- Nail Fungus (Onychomycosis)
- Ringworm (Tinea Corporis)
- Cutaneous Larva Migrans
- Chickenpox (Varicella)
- Shingles (Herpes Zoster)
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

SKINNET_CLASSES = [
    "Cellulitis",
    "Impetigo",
    "Athlete-foot",
    "Nail-fungus",
    "Ringworm",
    "Cutaneous-larva-migrans",
    "Chickenpox",
    "Shingles"
]

# Clinical mappings and human-friendly display names
DISPLAY_NAMES = {
    "Cellulitis": "Cellulitis (Acute Bacterial Dermic Infection)",
    "Impetigo": "Impetigo (Contagious Pyoderma)",
    "Athlete-foot": "Athlete's Foot (Tinea Pedis)",
    "Nail-fungus": "Nail Fungus (Onychomycosis)",
    "Ringworm": "Ringworm (Tinea Corporis)",
    "Cutaneous-larva-migrans": "Cutaneous Larva Migrans (Parasitic Creeping Eruption)",
    "Chickenpox": "Chickenpox (Varicella Zoster)",
    "Shingles": "Shingles (Herpes Zoster)"
}

TRIAGE_MAPPING = {
    "Cellulitis": {"level": "URGENT", "action": "Seek prompt medical care for antibiotic evaluation."},
    "Shingles": {"level": "URGENT", "action": "Seek prompt medical care; antivirals work best within 72 hours."},
    "Cutaneous-larva-migrans": {"level": "ROUTINE", "action": "Consult general physician/dermatologist for anthelmintic therapy."},
    "Impetigo": {"level": "ROUTINE", "action": "Consult healthcare clinic for topical/oral antibiotic treatment."},
    "Ringworm": {"level": "SELF_CARE", "action": "Initiate over-the-counter topical antifungal; consult if refractory."},
    "Athlete-foot": {"level": "SELF_CARE", "action": "Keep feet dry; use topical antifungal cream/powder."},
    "Nail-fungus": {"level": "ROUTINE", "action": "Consult dermatologist or podiatrist for prolonged antimycotic regimen."},
    "Chickenpox": {"level": "ROUTINE", "action": "Isolate from high-risk persons; symptomatic relief; monitor closely."}
}


class SkinNetEnsemble:
    """
    3-CNN Ensemble for multi-pathology infectious skin condition detection.
    """
    def __init__(self, models_dir: Optional[Union[str, Path]] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.classes = SKINNET_CLASSES

        if models_dir is None:
            # Look in standard locations
            candidate_dirs = [
                Path(__file__).resolve().parent.parent.parent / "models" / "skinnet",
                Path(__file__).resolve().parent.parent.parent.parent / "models" / "skinnet",
                Path("c:/Medha/backend/models/skinnet")
            ]
            for c in candidate_dirs:
                if (c / "efficientnet.pth").exists():
                    self.models_dir = c
                    break
            else:
                self.models_dir = candidate_dirs[0]
        else:
            self.models_dir = Path(models_dir)

        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._load_models()

    def _load_weights(self, model: nn.Module, filename: str) -> nn.Module:
        path = self.models_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing weights file at {path}")
        state = torch.load(str(path), map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def _load_models(self):
        num_classes = len(self.classes)

        # 1. EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier[1] = nn.Linear(1280, num_classes)
        self.efficientnet = self._load_weights(self.efficientnet, "efficientnet.pth")

        # 2. ResNet-18
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(512, num_classes)
        self.resnet = self._load_weights(self.resnet, "resnet.pth")

        # 3. MobileNetV2
        self.mobilenet = models.mobilenet_v2(weights=None)
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes)
        self.mobilenet = self._load_weights(self.mobilenet, "mobilenet.pth")

    def preprocess(self, img: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            # If BGR from OpenCV, convert to RGB
            if img.ndim == 3 and img.shape[2] == 3:
                img_rgb = img[:, :, ::-1]
            else:
                img_rgb = img
            pil_img = Image.fromarray(img_rgb)
        else:
            pil_img = img.convert("RGB")

        tensor = self._transform(pil_img)
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, img: Union[np.ndarray, Image.Image]) -> Dict:
        """
        Runs 3-CNN ensemble forward pass and returns top predictions with probabilities.
        """
        t0 = time.time()
        tensor = self.preprocess(img)

        with torch.no_grad():
            out1 = self.efficientnet(tensor)
            out2 = self.resnet(tensor)
            out3 = self.mobilenet(tensor)

            # Average logits across the 3 networks
            ensemble_logits = (out1 + out2 + out3) / 3.0
            probs_t = torch.softmax(ensemble_logits, dim=1).squeeze(0)
            probs = probs_t.cpu().numpy().tolist()

        latency_ms = round((time.time() - t0) * 1000, 1)

        # Ranked list of (class_name, prob)
        ranked = sorted(
            [(self.classes[i], float(probs[i])) for i in range(len(self.classes))],
            key=lambda x: x[1],
            reverse=True
        )

        top1_name, top1_prob = ranked[0]
        top3 = [
            {
                "disease": name,
                "display_name": DISPLAY_NAMES.get(name, name),
                "probability": round(prob, 4),
                "confidence_percent": round(prob * 100, 1),
                "rank": i + 1
            }
            for i, (name, prob) in enumerate(ranked[:3])
        ]

        triage_info = TRIAGE_MAPPING.get(top1_name, {"level": "ROUTINE", "action": "Consult dermatologist."})

        # Uncertainty threshold check
        is_uncertain = top1_prob < 0.35

        return {
            "engine": "SkinNet-3CNN-Ensemble",
            "condition": top1_name,
            "display_name": DISPLAY_NAMES.get(top1_name, top1_name),
            "confidence": round(top1_prob, 4),
            "confidence_percent": round(top1_prob * 100, 1),
            "is_uncertain": is_uncertain,
            "uncertainty_note": (
                "Confidence is under 35%. Symptom verification is strongly recommended to confirm the diagnosis."
                if is_uncertain else None
            ),
            "triage_level": triage_info["level"],
            "recommended_action": triage_info["action"],
            "top3": top3,
            "probabilities": {name: round(probs[i], 4) for i, name in enumerate(self.classes)},
            "latency_ms": latency_ms
        }
