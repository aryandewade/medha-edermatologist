# E-Dermatologist: Skin Disease Screening System

**Hackathon:** Somaiya Vidyavihar University, K J Somaiya School of Engineering  
**Clinical Partner:** Kaushalya Hospital  
**Document Type:** Problem statement and project brief (Mobile Spacer Edition)

---

## 1. Problem Statement

Access to dermatologists is severely restricted across rural and semi-urban India. Frontline community health workers (such as ASHA workers and local nurses) encounter high volumes of unmanaged or misidentified skin conditions, while patients often travel vast distances or delay care for months until manageable conditions worsen or malignant lesions advance.

Build a low-cost, mobile-first **AI skin disease screening and clinical triage system** that pairs standard **smartphones with an inexpensive 3D-printed optical spacer**. A health worker attaches the spacer to their phone camera, places it flush against the patient's skin, and captures a standardized close-up photo. The system checks capture sharpness, normalizes illumination and colour, localizes the lesion, and reports the likely condition category, confidence score, and top-3 differential breakdown to facilitate immediate primary care management or expedited referral to a dermatologist.

---

## 2. Objectives

1. Enable **instant mobile browser screening** on standard smartphones via a responsive Progressive Web App over secure HTTPS.
2. Standardize photographic capture using an accessible, **3D-printed focal spacer** that eliminates hand tremor focus shifts and standardizes distance.
3. Classify skin images across **common outpatient conditions** (Eczema, Psoriasis, Tinea, Acne), **Healthy Skin**, and a dedicated **Suspicious Lesion** triage class.
4. Normalize real-world lighting variations using automated **colour constancy** and reference patch calibration.
5. Provide **honest clinical triage guidance**: flag suspicious lesions for urgent referral and abstain on ambiguous or low-confidence cases.
6. Serve as an accessible **screening and triage decision aid**, explicitly complementing rather than replacing medical doctors.

---

## 3. Supported Clinical Classes

| Category | Conditions Included | Primary Triage Recommendation |
|---|---|---|
| **Inflammatory** | Eczema (Atopic Dermatitis), Psoriasis, Acne Vulgaris | Primary clinic management / Symptom monitoring |
| **Fungal Infection** | Tinea (Ringworm / Dermatophytosis) | Antifungal treatment / Hygiene guidance |
| **Normal Baseline** | Healthy Skin | Reassurance |
| **High-Risk Lesions** | Suspicious Neoplastic Lesions (Melanoma, BCC, SCC, Actinic Keratosis) | **Urgent Dermatologist Referral & In-Person Review** |

*(Future Phase: Specialized Dermatoscope Modality for fine-grained sub-classification of melanocytic and non-melanocytic lesions).*

---

## 4. Scope

**In Scope:**
- Mobile-first web interface (PWA) with live viewfinder framing guide.
- Standardized capture with 3D-printed spacer and reference colour patch.
- Secure HTTPS tunneling (Cloudflare Tunnel) connecting mobile phones to the backend.
- Backend image processing pipeline: blur/glare quality gate, colour constancy, lesion localization.
- Multi-class classification via fine-tuned EfficientNet-B0.
- Confidence calibration (Temperature Scaling) and urgent referral rules.
- Morphological explainability via Grad-CAM++ activation heatmaps.

**Out of Scope (Hackathon Prototype):**
- Definitive clinical diagnosis or automated prescription dispensing.
- Invasive cancer staging or replacing dermatopathological biopsy.
- Long-term Electronic Medical Record (EMR) persistent storage.
- Ultra-rare skin pathologies outside the supported taxonomy.

---

## 5. Hardware Specifications

| Item | Specification | Status |
|---|---|---|
| **Smartphone** | Standard rear autofocus camera (e.g. 12MP+ sensor, iOS or Android) | Required (Primary) |
| **3D-Printed Spacer** | Fixed 35 mm focal height, matte-black non-reflective interior, phone clip | Required (Primary) |
| **Reference Patch** | Dual 18% neutral gray & 90% white patch with 5 mm scale bar | Required (Primary) |
| **Hygiene Supplies** | 70% isopropyl alcohol sanitization wipes for spacer contact rim | Required |
| **Laptop / Compute** | 8 GB+ RAM for running backend inference (CPU inference via ONNX) | Required |
| **Network Ingress** | Cloudflare Tunnel or ngrok providing public HTTPS endpoint | Required |
| **USB Dermatoscope** | Handheld UVC microscope camera (for secondary multimodal phase) | Phase 2 Extension |

---

## 6. Software Stack

- **Frontend:** React, Vite, Tailwind CSS, HTML5 Canvas API, Mobile WebRTC (`getUserMedia`).
- **Networking:** Cloudflare Tunnel (`cloudflared`) / ngrok for mobile HTTPS camera access.
- **Backend:** FastAPI (Python 3.10+), Uvicorn, Pydantic v2.
- **Image Processing & Vision:** OpenCV (`cv2`), NumPy, SciPy (colour constancy, Otsu segmentation, Laplacian sharpness).
- **ML Framework & Serving:** PyTorch, Torchvision, ONNX Runtime (CPU-optimized), Albumentations.

---

## 7. System Workflow

```
[Smartphone + 3D Spacer] ──(HTTPS)──> [FastAPI Backend]
                                             │
                                  ├─ 1. Quality Validation (Blur, light, glare)
                                  ├─ 2. Colour Constancy (Shades of Gray)
                                  ├─ 3. Lesion Localization (ROI square crop)
                                  ├─ 4. EfficientNet-B0 ONNX Inference
                                  ├─ 5. Calibrated Triage Rules
                                  └─ 6. Grad-CAM++ Morphological Overlay
                                             │
                                      (JSON Result)
                                             ▼
                             [Mobile Results & Referral View]
```

---

## 8. Disclaimer
This software is a screening decision-support aid and does not constitute a definitive medical diagnosis. All patients presenting with persistent, worsening, or suspicious skin lesions must be referred to a qualified dermatologist.
