"""
API Router for Skin Disease Predictions, System Health, and PDF Referral Reports
"""

import sys
import os
import uuid
import base64
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Body
from fastapi.responses import Response

# Ensure ml/src and services are importable
APP_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = APP_DIR.parent.parent
ML_SRC = BASE_DIR / "ml" / "src"

for p in [str(APP_DIR), str(ML_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import config
from schemas import (
    HealthResponse,
    ClassesResponse,
    ClassItem,
    PredictResponse,
    PredictionItem,
    Top3Item,
    AdviceItem,
    QualityItem
)
from inference import MobileSpacerInferenceEngine
from services.report import generate_referral_pdf

router = APIRouter(prefix="/api/v1", tags=["Inference"])

# Global inference engine singleton
_engine: Optional[MobileSpacerInferenceEngine] = None
_warmed_up = False


def get_engine() -> MobileSpacerInferenceEngine:
    global _engine, _warmed_up
    if _engine is None:
        model_checkpoint = os.getenv("MODEL_PATH", None)
        if not model_checkpoint:
            for cand in [BASE_DIR / "backend" / "models" / "best_model.pt", BASE_DIR / "models" / "best_model.pt"]:
                if cand.exists():
                    model_checkpoint = str(cand)
                    break
        _engine = MobileSpacerInferenceEngine(
            checkpoint_path=model_checkpoint,
            device="cpu",
            uncertainty_threshold=config.thresholds.get("uncertain_max_prob", 0.50),
            urgent_suspicious_threshold=config.thresholds.get("urgent_suspicious_prob", 0.25)
        )
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        _ = _engine.predict_image(dummy)
        _warmed_up = True
        print("[Router] Inference engine loaded and warmed up.")
    return _engine


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Returns backend and model status."""
    engine = get_engine()
    return HealthResponse(
        status="ok",
        model_version=config.version,
        active_modality=config.modality,
        num_classes=len(config.classes),
        warmed_up=_warmed_up
    )


@router.get("/classes", response_model=ClassesResponse)
def get_classes():
    """Returns supported clinical condition classes."""
    items = [ClassItem(**c) for c in config.classes]
    return ClassesResponse(classes=items)


@router.post("/predict", response_model=PredictResponse)
async def predict_image(
    image: UploadFile = File(...),
    modality: Optional[str] = Form("mobile_spacer"),
    device_info: Optional[str] = Form(None),
    capture_height_mm: Optional[float] = Form(35.0),
    heatmap: Optional[bool] = Form(False),
    lang: Optional[str] = Form("en")
):
    """
    Analyzes skin photograph from mobile smartphone spacer.
    Runs quality checks, colour constancy, lesion localization, and calibrated inference.
    """
    if image.content_type not in ["image/jpeg", "image/png", "image/webp", "application/octet-stream"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {image.content_type}. Expected JPEG, PNG, or WEBP."
        )

    contents = await image.read()
    if len(contents) > 8 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 8 MB limit."
        )

    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image binary. Ensure the file is a valid image."
        )

    engine = get_engine()
    result = engine.predict_image(img_bgr)

    heatmap_base64 = None
    if heatmap and "roi_preview" in result:
        try:
            roi_rgb = result["roi_preview"]
            gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
            heatmap_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR), 0.6, heatmap_img, 0.4, 0)
            _, buffer = cv2.imencode(".png", overlay)
            heatmap_base64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
        except Exception as e:
            print(f"[Warning] Failed to generate heatmap: {e}")

    req_id = str(uuid.uuid4())[:8]

    return PredictResponse(
        request_id=req_id,
        modality=modality or "mobile_spacer",
        prediction=PredictionItem(**result["prediction"]),
        top3=[Top3Item(**item) for item in result["top3"]],
        probabilities=result["probabilities"],
        advice=AdviceItem(**result["advice"]),
        quality=QualityItem(
            blur_score=result["quality"]["blur_score"],
            brightness=result["quality"]["brightness"],
            glare_ratio=result["quality"]["glare_ratio"],
            reference_patch_found=result["quality"].get("reference_patch_found", False),
            colour_cast=result["quality"].get("colour_cast", "normal"),
            status=result["quality"]["status"],
            issues=result["quality"]["issues"]
        ),
        heatmap_png_base64=heatmap_base64,
        model_version=config.version,
        latency_ms=result["latency_ms"]
    )


@router.post("/report")
async def generate_report(
    payload: Dict[str, Any] = Body(...)
):
    """
    Generates a 1-page PDF referral summary card from screening results.
    """
    result_data = payload.get("result", payload)
    patient_ref = payload.get("patient_ref", "Field Screening Case")
    notes = payload.get("notes", None)
    image_base64 = payload.get("image_base64", None)

    try:
        pdf_bytes = generate_referral_pdf(
            result_data=result_data,
            patient_ref=patient_ref,
            notes=notes,
            image_base64=image_base64
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=EDerm_Referral_{result_data.get('request_id', 'case')}.pdf"
            }
        )
    except Exception as e:
        print(f"[Error] PDF generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF report: {str(e)}"
        )
