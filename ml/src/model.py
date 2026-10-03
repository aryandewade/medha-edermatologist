"""
Skin Disease Classifier using EfficientNet-B0 (Mobile Spacer Profile)
Architecture: EfficientNet-B0 (ImageNet-pretrained) with custom classification head
Target Classes (6): Eczema, Psoriasis, Tinea, Acne, Healthy Skin, Suspicious Lesion
"""

from typing import Optional, List
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ResidualDermClassifierHead(nn.Module):
    """
    V2 High-Performance Residual Clinical Head for Dermatological Classification.
    Combines:
      1. Direct linear projection from 1280-d embeddings (preserves global linear separation).
      2. Multi-stage non-linear MLP with LayerNorm, GELU, and Dropout (separates non-linearly
         overlapping erythematous and papulosquamous conditions like eczema vs psoriasis vs tinea).
    """

    def __init__(
        self,
        in_features: int = 1280,
        num_classes: int = 6,
        hidden_dim: int = 384,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.skip = nn.Linear(in_features, num_classes)
        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate * 0.75),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.mlp(x)


class SkinDiseaseEfficientNetB0(nn.Module):
    """
    EfficientNet-B0 model adapted for 6-class mobile skin screening.
    
    Workflow:
      Input (B, 3, 224, 224) 
        -> EfficientNet-B0 Feature Extractor (MBConv blocks)
        -> AdaptiveAvgPool2d(1) -> 1280 dim embedding
        -> Head (V1: Dropout+Linear, or V2: Residual Clinical Head)
        -> Logits (B, 6)
    """

    DEFAULT_CLASSES = [
        "eczema", 
        "psoriasis", 
        "tinea", 
        "acne", 
        "healthy", 
        "suspicious_lesion"
    ]

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        class_names: Optional[List[str]] = None,
        head_type: str = "linear"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or self.DEFAULT_CLASSES
        self.head_type = head_type

        # 1. Load pretrained EfficientNet-B0
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Extract input features to classifier (1280 for EfficientNet-B0)
        in_features = self.backbone.classifier[1].in_features

        # 3. Custom classification head
        if head_type == "residual_v2":
            self.backbone.classifier = ResidualDermClassifierHead(
                in_features=in_features,
                num_classes=num_classes,
                dropout_rate=dropout_rate
            )
        else:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features=in_features, out_features=num_classes)
            )

        # Default temperature parameter for post-hoc calibration
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw class logits.
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate 1280-dimensional feature embeddings."""
        feats = self.backbone.features(x)
        pooled = self.backbone.avgpool(feats)
        return torch.flatten(pooled, 1)

    def predict_proba(
        self, 
        x: torch.Tensor, 
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute calibrated softmax probabilities.
        """
        T = temperature if temperature is not None else self.temperature.item()
        logits = self.forward(x)
        scaled_logits = logits / max(T, 1e-4)
        return torch.softmax(scaled_logits, dim=-1)

    def freeze_backbone(self) -> None:
        """Freeze all backbone feature layers for initial transfer learning warmup."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("[Model] Backbone features frozen. Only classifier head is trainable.")

    def unfreeze_backbone(self, from_stage: Optional[int] = None) -> None:
        """Unfreeze backbone layers for fine-tuning."""
        if from_stage is None:
            for param in self.backbone.features.parameters():
                param.requires_grad = True
            print("[Model] All backbone features unfrozen for end-to-end fine-tuning.")
        else:
            for i, layer in enumerate(self.backbone.features):
                requires_grad = i >= from_stage
                for param in layer.parameters():
                    param.requires_grad = requires_grad
            print(f"[Model] Backbone features from stage {from_stage} onwards unfrozen.")

    def get_cam_target_layer(self) -> nn.Module:
        """Returns the final convolutional layer for Grad-CAM++ visualization."""
        return self.backbone.features[-1]


def build_model(
    num_classes: int = 6,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    head_type: Optional[str] = None,
    strict: bool = True
) -> SkinDiseaseEfficientNetB0:
    """Helper factory to instantiate and optionally load weights with auto-detection of V1 vs V2 heads."""
    resolved_head = head_type or "linear"
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        if head_type is None:
            if "backbone.classifier.skip.weight" in state_dict or (isinstance(checkpoint, dict) and checkpoint.get("version") == "v2"):
                resolved_head = "residual_v2"
            else:
                resolved_head = "linear"

    model = SkinDiseaseEfficientNetB0(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        head_type=resolved_head
    )
    if checkpoint:
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=strict)
        print(f"[Model] Loaded weights ({resolved_head}) from {checkpoint_path}")
    
    model.to(device)
    return model
