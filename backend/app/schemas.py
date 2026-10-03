"""
Pydantic Schemas for E-Dermatologist REST API
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str
    active_modality: str
    num_classes: int
    warmed_up: bool


class ClassItem(BaseModel):
    id: str
    label: str
    name: Optional[str] = None
    severity: str
    color: str
    description: Optional[str] = None


class ClassesResponse(BaseModel):
    classes: List[ClassItem]


class PredictionItem(BaseModel):
    class_id: str
    label: str
    confidence: float
    severity: str


class Top3Item(BaseModel):
    class_id: str
    label: str
    probability: float


class AdviceItem(BaseModel):
    level: str  # "ok" | "uncertain" | "urgent" | "out_of_scope"
    title: str
    message: str


class QualityItem(BaseModel):
    blur_score: float
    brightness: float
    glare_ratio: float
    reference_patch_found: bool = False
    colour_cast: str = "normal"
    status: str  # "good" | "warning" | "rejected"
    issues: List[str] = []


class PredictResponse(BaseModel):
    request_id: str
    modality: str
    prediction: PredictionItem
    top3: List[Top3Item]
    probabilities: Dict[str, float]
    advice: AdviceItem
    quality: QualityItem
    heatmap_png_base64: Optional[str] = None
    model_version: str
    latency_ms: float
