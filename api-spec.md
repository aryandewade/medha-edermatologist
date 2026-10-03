# API Specification (v2.0 Mobile-First)

Base URL: `http://localhost:8000` (or `https://<tunnel-subdomain>.trycloudflare.com`)  
Format: JSON, except `/predict` request (`multipart/form-data`).  
Prefix: `/api/v1`

---

## 1. `GET /api/v1/health`
Verifies backend operational status, model readiness, and active modality.

**Response (200 OK):**
```json
{
  "status": "ok",
  "model_version": "v2.0-mobile-effnetb0",
  "active_modality": "mobile_spacer",
  "num_classes": 6,
  "warmed_up": true
}
```

---

## 2. `GET /api/v1/classes`
Retrieves list of supported clinical conditions, display labels, severity categories, and UI color tokens.

**Response (200 OK):**
```json
{
  "classes": [
    {"id": "eczema", "label": "Eczema", "severity": "common", "color": "#2563EB"},
    {"id": "psoriasis", "label": "Psoriasis", "severity": "common", "color": "#D97706"},
    {"id": "tinea", "label": "Tinea (Ringworm)", "severity": "common", "color": "#7C3AED"},
    {"id": "acne", "label": "Acne", "severity": "common", "color": "#059669"},
    {"id": "healthy", "label": "Healthy Skin", "severity": "normal", "color": "#10B981"},
    {"id": "suspicious_lesion", "label": "Suspicious Lesion", "severity": "urgent", "color": "#DC2626"}
  ]
}
```

---

## 3. `POST /api/v1/predict`
Analyzes a skin photograph captured via mobile smartphone with the 3D-printed spacer.

**Request (`multipart/form-data`):**
| Field | Type | Required | Description |
|---|---|---|---|
| `image` | Binary (JPEG/PNG) | Yes | Full-resolution captured frame (Max 8 MB) |
| `modality` | String | No | `"mobile_spacer"` (default) or `"dermatoscope"` |
| `device_info` | String | No | Client phone model/OS (e.g. `"Samsung_Galaxy_S23_Chrome"`) |
| `capture_height_mm` | Float | No | Focal spacer height in mm (default `35.0`) |
| `heatmap` | Boolean | No | Include base64 Grad-CAM++ heatmap overlay (default `false`) |
| `lang` | String | No | Localized advice text: `"en"` (default), `"hi"`, `"mr"` |

**Response (200 OK) — Confident Finding:**
```json
{
  "request_id": "c7a8b9e1-45f2",
  "modality": "mobile_spacer",
  "prediction": {
    "class_id": "eczema",
    "label": "Eczema",
    "confidence": 0.842,
    "severity": "common"
  },
  "top3": [
    {"class_id": "eczema", "label": "Eczema", "probability": 0.842},
    {"class_id": "psoriasis", "label": "Psoriasis", "probability": 0.098},
    {"class_id": "tinea", "label": "Tinea (Ringworm)", "probability": 0.035}
  ],
  "probabilities": {
    "eczema": 0.842,
    "psoriasis": 0.098,
    "tinea": 0.035,
    "acne": 0.012,
    "healthy": 0.008,
    "suspicious_lesion": 0.005
  },
  "advice": {
    "level": "ok",
    "title": "Features Consistent with Eczema",
    "message": "Presentation aligns with inflammatory eczema. Standard primary care management recommended."
  },
  "quality": {
    "blur_score": 138.4,
    "brightness": 124.0,
    "glare_ratio": 0.02,
    "reference_patch_found": true,
    "colour_cast": "corrected_neutral",
    "status": "good"
  },
  "heatmap_png_base64": null,
  "model_version": "v2.0-mobile-effnetb0",
  "latency_ms": 38.5
}
```

**Response (200 OK) — Urgent Referral Triage (Suspicious Lesion):**
```json
{
  "request_id": "e9f0d1a2-83b4",
  "modality": "mobile_spacer",
  "prediction": {
    "class_id": "suspicious_lesion",
    "label": "Suspicious Lesion",
    "confidence": 0.761,
    "severity": "urgent"
  },
  "top3": [
    {"class_id": "suspicious_lesion", "label": "Suspicious Lesion", "probability": 0.761},
    {"class_id": "psoriasis", "label": "Psoriasis", "probability": 0.145},
    {"class_id": "eczema", "label": "Eczema", "probability": 0.062}
  ],
  "advice": {
    "level": "urgent",
    "title": "Specialist Review Required",
    "message": "Morphological features warrant formal in-person examination by a dermatologist. Facilitate referral."
  },
  "quality": {
    "blur_score": 152.0,
    "brightness": 115.0,
    "glare_ratio": 0.01,
    "reference_patch_found": true,
    "colour_cast": "corrected_neutral",
    "status": "good"
  },
  "heatmap_png_base64": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "latency_ms": 42.0
}
```

---

## 4. Error Responses

```json
{
  "error": {
    "code": "image_too_blurry",
    "message": "Image is out of focus. Ensure the spacer is resting flat against the skin and hold phone steady.",
    "details": {
      "blur_score": 28.6,
      "min_required": 55.0
    }
  }
}
```

| HTTP Status | Error Code | Description / Remedy |
|---|---|---|
| **400** | `invalid_request` | Missing image payload or malformed multipart data. |
| **413** | `file_too_large` | File exceeds maximum upload limit ($> 8\text{ MB}$). |
| **415** | `unsupported_media_type` | File is not JPEG, PNG, or WEBP. |
| **422** | `image_too_blurry` | Focus below threshold ($< 55.0$). Hold phone steady. |
| **422** | `image_too_dark` | Inadequate ambient illumination ($< 40$). Increase room lighting. |
| **422** | `spacer_misaligned` | Severe uneven light leakage or background detected. |
| **503** | `model_not_ready` | Model weights still loading or undergoing warmup. |
