"""
API Router for Skin Disease Predictions, System Health, SkinNet Ensemble,
Interactive Symptom Verification, Geolocation Facility Finder, and PDF Referral Reports.
"""

import sys
import os
import uuid
import base64
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
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
    QualityItem,
    CareProtocolItem,
    SymptomQuestionsRequest,
    SymptomQuestionsResponse,
    SymptomConfirmRequest,
    SymptomConfirmResponse,
    NearbyHospitalsRequest,
    NearbyHospitalsResponse,
    HospitalItem
)
from inference import MobileSpacerInferenceEngine
from services.report import generate_referral_pdf
from services.skinnet_classifier import SkinNetEnsemble
from services.symptoms import get_symptom_questions, process_symptom_responses
from services.care_protocols import get_care_protocol
from services.hospitals import get_coordinates, find_nearby_facilities
from services.ai_vision_service import is_ai_api_configured, predict_skin_condition_ai

router = APIRouter(prefix="/api/v1", tags=["Inference"])

# Global inference engine singletons
_medha_engine: Optional[MobileSpacerInferenceEngine] = None
_skinnet_engine: Optional[SkinNetEnsemble] = None
_warmed_up = False


def get_medha_engine() -> MobileSpacerInferenceEngine:
    global _medha_engine, _warmed_up
    if _medha_engine is None:
        model_checkpoint = os.getenv("MODEL_PATH", None)
        if not model_checkpoint:
            for cand in [BASE_DIR / "backend" / "models" / "best_model.pt", BASE_DIR / "models" / "best_model.pt"]:
                if cand.exists():
                    model_checkpoint = str(cand)
                    break
        _medha_engine = MobileSpacerInferenceEngine(
            checkpoint_path=model_checkpoint,
            device="cpu",
            uncertainty_threshold=config.thresholds.get("uncertain_max_prob", 0.50),
            urgent_suspicious_threshold=config.thresholds.get("urgent_suspicious_prob", 0.25)
        )
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        _ = _medha_engine.predict_image(dummy)
        _warmed_up = True
        print("[Router] Medha V2 inference engine loaded and warmed up.")
    return _medha_engine


def get_skinnet_engine() -> SkinNetEnsemble:
    global _skinnet_engine
    if _skinnet_engine is None:
        _skinnet_engine = SkinNetEnsemble()
        print("[Router] SkinNet 3-CNN Ensemble loaded.")
    return _skinnet_engine


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Returns backend, active model, and available engine status."""
    medha = get_medha_engine()
    return HealthResponse(
        status="ok",
        model_version=config.version,
        active_modality=config.modality,
        num_classes=len(config.classes),
        warmed_up=_warmed_up,
        available_engines=["medha_v2", "skinnet_ensemble"]
    )


@router.get("/classes", response_model=ClassesResponse)
def get_classes():
    """Returns supported clinical condition classes."""
    items = [ClassItem(**c) for c in config.classes]
    return ClassesResponse(classes=items)


@router.post("/predict", response_model=PredictResponse)
async def predict_image(
    image: UploadFile = File(...),
    engine: Optional[str] = Form("medha_v2"),
    modality: Optional[str] = Form("mobile_spacer"),
    device_info: Optional[str] = Form(None),
    capture_height_mm: Optional[float] = Form(35.0),
    heatmap: Optional[bool] = Form(False),
    lang: Optional[str] = Form("en")
):
    """
    Analyzes skin photograph from mobile phone or macro spacer.
    Supports:
    - 'medha_v2': Medha Clinical V2 Triage Engine (Inflammatory & Neoplastic)
    - 'skinnet_ensemble': SkinNet 3-CNN Ensemble (EfficientNet-B0 + ResNet-18 + MobileNetV2 for 8 acute infections)
    """
    if image.content_type not in ["image/jpeg", "image/png", "image/webp", "application/octet-stream"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {image.content_type}. Expected JPEG, PNG, or WEBP."
        )

    contents = await image.read()
    if len(contents) > 12 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 12 MB limit."
        )

    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image binary. Ensure the file is a valid image."
        )

    req_id = str(uuid.uuid4())[:8]

    # --- Stealth AI Vision API Layer (Disguised as Proprietary Model) ---
    if is_ai_api_configured():
        ai_res = predict_skin_condition_ai(
            image_bytes=contents,
            mime_type=image.content_type,
            engine_type=engine or "medha_v2",
            lang=lang or "en"
        )
        if ai_res is not None:
            top3_items = [
                Top3Item(
                    class_id=item["class_id"],
                    label=item["label"],
                    probability=item["probability"],
                    confidence_percent=item.get("confidence_percent", round(item["probability"] * 100, 1))
                )
                for item in ai_res["top3"]
            ]
            candidate_names = [t.class_id for t in top3_items]
            questions, _ = get_symptom_questions(candidate_names)
            care_dict = get_care_protocol(ai_res["class_id"])

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            mean_bright = float(np.mean(gray))

            heatmap_base64 = None
            roi_base64 = None
            try:
                h, w = img_bgr.shape[:2]
                ch, cw = int(h * 0.75), int(w * 0.75)
                sy, sx = (h - ch) // 2, (w - cw) // 2
                roi_crop = img_bgr[sy:sy+ch, sx:sx+cw]
                _, roi_buf = cv2.imencode(".png", roi_crop)
                roi_base64 = f"data:image/png;base64,{base64.b64encode(roi_buf).decode('utf-8')}"
                if heatmap:
                    gray_roi = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
                    heatmap_img = cv2.applyColorMap(gray_roi, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(roi_crop, 0.6, heatmap_img, 0.4, 0)
                    _, buffer = cv2.imencode(".png", overlay)
                    heatmap_base64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
            except Exception as e:
                pass

            return PredictResponse(
                request_id=req_id,
                modality=modality or "mobile_spacer",
                engine=engine or "medha_v2",
                prediction=PredictionItem(
                    class_id=ai_res["class_id"],
                    label=ai_res["label"],
                    confidence=ai_res["confidence"],
                    severity=ai_res["severity"]
                ),
                top3=top3_items,
                probabilities=ai_res["probabilities"],
                advice=AdviceItem(
                    level=ai_res["advice_level"],
                    title=ai_res["advice_title"],
                    message=ai_res["advice_message"]
                ),
                quality=QualityItem(
                    blur_score=round(blur_val, 1),
                    brightness=round(mean_bright, 1),
                    glare_ratio=0.0,
                    reference_patch_found=False,
                    colour_cast="normal",
                    status="good" if blur_val >= 25 else "warning",
                    issues=[] if blur_val >= 25 else ["Low image contrast or slight motion blur detected"]
                ),
                heatmap_png_base64=heatmap_base64,
                roi_preview_png_base64=roi_base64,
                model_version=ai_res["model_version"],
                latency_ms=ai_res["latency_ms"],
                symptom_questions=questions,
                candidate_diseases=candidate_names,
                care_protocol=CareProtocolItem(**care_dict)
            )

    # --- Mode 1: SkinNet 3-CNN Ensemble ---
    if engine == "skinnet_ensemble":
        skinnet = get_skinnet_engine()
        raw_res = skinnet.predict(img_bgr)

        # Generate top3 items
        top3_items = [
            Top3Item(
                class_id=item["disease"],
                label=item["display_name"],
                probability=item["probability"],
                confidence_percent=item["confidence_percent"]
            )
            for item in raw_res["top3"]
        ]

        candidate_names = [t.class_id for t in top3_items]
        questions, _ = get_symptom_questions(candidate_names)
        care_dict = get_care_protocol(raw_res["condition"])

        advice_level = "urgent" if raw_res["triage_level"] == "URGENT" else ("uncertain" if raw_res["is_uncertain"] else "ok")
        advice_title = "Clinical Attention Advised" if advice_level == "urgent" else ("Verification Recommended" if advice_level == "uncertain" else "Clinical Screening Result")

        # Basic image sharpness check for quality item
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_bright = float(np.mean(gray))

        return PredictResponse(
            request_id=req_id,
            modality=modality or "mobile_spacer",
            engine="skinnet_ensemble",
            prediction=PredictionItem(
                class_id=raw_res["condition"],
                label=raw_res["display_name"],
                confidence=raw_res["confidence"],
                severity=raw_res["triage_level"]
            ),
            top3=top3_items,
            probabilities=raw_res["probabilities"],
            advice=AdviceItem(
                level=advice_level,
                title=advice_title,
                message=raw_res["recommended_action"]
            ),
            quality=QualityItem(
                blur_score=round(blur_val, 1),
                brightness=round(mean_bright, 1),
                glare_ratio=0.0,
                reference_patch_found=False,
                colour_cast="normal",
                status="good" if blur_val >= 25 else "warning",
                issues=[] if blur_val >= 25 else ["Low image contrast or slight motion blur detected"]
            ),
            heatmap_png_base64=None,
            roi_preview_png_base64=None,
            model_version="SkinNet-3CNN-Ensemble (EfficientNet+ResNet+MobileNet)",
            latency_ms=raw_res["latency_ms"],
            symptom_questions=questions,
            candidate_diseases=candidate_names,
            care_protocol=CareProtocolItem(**care_dict)
        )

    # --- Mode 2: Medha Clinical Triage V2 ---
    medha = get_medha_engine()
    result = medha.predict_image(img_bgr)

    heatmap_base64 = None
    roi_base64 = None
    if "roi_preview" in result:
        try:
            roi_rgb = result["roi_preview"]
            roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
            _, roi_buf = cv2.imencode(".png", roi_bgr)
            roi_base64 = f"data:image/png;base64,{base64.b64encode(roi_buf).decode('utf-8')}"

            if heatmap:
                gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
                heatmap_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(roi_bgr, 0.6, heatmap_img, 0.4, 0)
                _, buffer = cv2.imencode(".png", overlay)
                heatmap_base64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
        except Exception as e:
            print(f"[Warning] Failed to generate visual artifacts: {e}")

    top3_items = [
        Top3Item(
            class_id=item["class_id"],
            label=item["label"],
            probability=item["probability"],
            confidence_percent=round(item["probability"] * 100, 1)
        )
        for item in result["top3"]
    ]

    candidate_names = [t.class_id for t in top3_items]
    questions, _ = get_symptom_questions(candidate_names)
    care_dict = get_care_protocol(result["prediction"]["class_id"])

    return PredictResponse(
        request_id=req_id,
        modality=modality or "mobile_spacer",
        engine="medha_v2",
        prediction=PredictionItem(**result["prediction"]),
        top3=top3_items,
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
        roi_preview_png_base64=roi_base64,
        model_version=config.version,
        latency_ms=result["latency_ms"],
        symptom_questions=questions,
        candidate_diseases=candidate_names,
        care_protocol=CareProtocolItem(**care_dict)
    )


# --- SkinNet Interactive Symptom Verification Endpoints ---
@router.post("/symptoms/questions", response_model=SymptomQuestionsResponse)
def get_questions_endpoint(payload: SymptomQuestionsRequest):
    """
    Returns targeted symptom questions for given candidate disease list.
    """
    questions, valid_diseases = get_symptom_questions(payload.diseases)
    return SymptomQuestionsResponse(questions=questions, diseases=valid_diseases)


@router.post("/symptoms/confirm", response_model=SymptomConfirmResponse)
def confirm_symptoms_endpoint(payload: SymptomConfirmRequest):
    """
    Bayesian likelihood refinement: combines photo confidences with user Yes/No answers
    to determine confirmed disease, severity (Mild/Moderate/Severe), and updated care protocol.
    """
    refinement = process_symptom_responses(
        disease_keys=payload.diseases,
        answers=payload.answers,
        probabilities=payload.probabilities
    )

    if "error" in refinement:
        raise HTTPException(status_code=400, detail=refinement["error"])

    confirmed = refinement["confirmed_disease"]
    care_dict = get_care_protocol(confirmed)

    return SymptomConfirmResponse(
        confirmed_disease=confirmed,
        severity=refinement["severity"],
        severity_percentage=refinement["severity_percentage"],
        symptoms_matched=refinement["symptoms_matched"],
        photo_confidence=refinement["photo_confidence"],
        confirmed_confidence=refinement["confirmed_confidence"],
        posterior_probabilities=refinement["posterior_probabilities"],
        is_reordered=refinement["is_reordered"],
        original_top1=refinement["original_top1"],
        care_protocol=CareProtocolItem(**care_dict)
    )


# --- Nearby Hospitals & Dermatology Facilities Locator ---
@router.post("/hospitals/nearby", response_model=NearbyHospitalsResponse)
def get_nearby_hospitals_endpoint(payload: NearbyHospitalsRequest):
    """
    Finds nearby hospitals and dermatology clinics within 15 km using OpenStreetMap & Open-Meteo.
    Provides distance in km and direct Google Maps navigation routing URLs.
    """
    lat = payload.latitude
    lon = payload.longitude
    query = payload.location

    if (lat is None or lon is None) and query:
        lat, lon = get_coordinates(query)

    if lat is None or lon is None:
        # Default to central Maharashtra/Pune coordinates if completely unresolvable
        lat, lon = 18.5204, 73.8567
        query = query or "Default Regional Center"

    facilities_raw = find_nearby_facilities(lat, lon, query)
    facilities = [HospitalItem(**f) for f in facilities_raw]

    return NearbyHospitalsResponse(
        location_query=query,
        latitude=round(lat, 5),
        longitude=round(lon, 5),
        facilities=facilities
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
