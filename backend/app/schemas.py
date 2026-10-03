"""
Pydantic Schemas for E-Dermatologist REST API
Extended with SkinNet-Analyzer Symptom Refinement and Facility Locator schemas.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str
    active_modality: str
    num_classes: int
    warmed_up: bool
    available_engines: List[str] = ["medha_v2", "skinnet_ensemble"]


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
    confidence_percent: Optional[float] = None


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


class CareProtocolItem(BaseModel):
    external: List[str] = []
    internal: List[str] = []
    care: List[str] = []
    urgent: str = ""


class PredictResponse(BaseModel):
    request_id: str
    modality: str
    engine: str = "medha_v2"
    prediction: PredictionItem
    top3: List[Top3Item]
    probabilities: Dict[str, float]
    advice: AdviceItem
    quality: QualityItem
    heatmap_png_base64: Optional[str] = None
    roi_preview_png_base64: Optional[str] = None
    model_version: str
    latency_ms: float
    symptom_questions: Optional[Dict[str, str]] = None
    candidate_diseases: Optional[List[str]] = None
    care_protocol: Optional[CareProtocolItem] = None


# --- SkinNet Symptom Verification Schemas ---
class SymptomQuestionsRequest(BaseModel):
    diseases: List[str]


class SymptomQuestionsResponse(BaseModel):
    questions: Dict[str, str]
    diseases: List[str]


class SymptomConfirmRequest(BaseModel):
    diseases: List[str]
    answers: Dict[str, Any]  # "symptom_name" -> "1" / "0" / bool
    probabilities: Optional[List[float]] = None


class SymptomConfirmResponse(BaseModel):
    confirmed_disease: str
    severity: str  # "Mild" | "Moderate" | "Severe" | "Out of Class"
    severity_percentage: float
    symptoms_matched: str
    photo_confidence: float
    confirmed_confidence: float
    posterior_probabilities: Dict[str, float]
    is_reordered: bool
    original_top1: str
    care_protocol: CareProtocolItem


# --- Hospital Locator Schemas ---
class HospitalItem(BaseModel):
    name: str
    distance_km: float
    latitude: float
    longitude: float
    location_label: str
    maps_url: str


class NearbyHospitalsRequest(BaseModel):
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class NearbyHospitalsResponse(BaseModel):
    location_query: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    facilities: List[HospitalItem] = []
