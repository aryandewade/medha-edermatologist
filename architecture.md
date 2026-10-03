# System Architecture (Mobile-First)

## 1. Overview
The architecture is designed as a **mobile-first client-server system**. A smartphone running a Progressive Web App (PWA) captures skin images through a 3D-printed optical spacer. The image and capture metadata are transmitted securely over HTTPS to a FastAPI inference backend that executes automated quality checks, colour constancy, lesion localization, model inference, and clinical triage rules.

```
+---------------------------------------+
|  Smartphone + 3D-Printed Spacer       |
|  Mobile Browser (React + Vite PWA)    |
|  - Rear camera (facingMode: environ)  |
|  - Live circular reticle & guide      |
|  - Canvas frame snapshot              |
+---------------------------------------+
                   │
                   │ POST /api/v1/predict
                   │ (Multipart JPEG + modality metadata)
                   │ Over HTTPS (Cloudflare Tunnel / ngrok / Cloud)
                   ▼
+-------------------------------------------------------+
|  FastAPI Backend Server                               |
|                                                       |
|  1. Ingestion & Validation (file type, size, EXIF)   |
|  2. Quality Gate (Laplacian blur, lighting, glare)    |
|  3. Colour Normalization (Gray-World / Patch balance) |
|  4. Lesion Localization (ROI crop & center-framing)   |
|  5. Preprocessing (Resize 224×224, ImageNet norm)     |
|  6. Inference Engine (ONNX Runtime / PyTorch)         |
|  7. Calibration & Uncertainty (T-scaling, OOD)        |
|  8. Clinical Triage Rules (Referral / Abstention)     |
|  9. Explainability (Grad-CAM++ heatmap overlay)       |
+-------------------------------------------------------+
                   │
                   │ JSON Result + Base64 Heatmap
                   ▼
+---------------------------------------+
|  Mobile Results View                  |
|  - Condition card & confidence score  |
|  - Top-3 differential breakdown       |
|  - Triage Referral Banner             |
|  - Heatmap visual overlay             |
+---------------------------------------+
```

---

## 2. Components & Responsibilities

| Component | Technology | Primary Responsibility |
|---|---|---|
| **Mobile Web Client** | React, Vite, Tailwind CSS, Lucide Icons | Touch-friendly camera feed, spacer framing guide, one-thumb capture, instant results display. |
| **Secure Ingress / Tunnel** | Cloudflare Tunnel / ngrok / Caddy HTTPS | Provides valid public SSL/TLS certificate required by mobile browsers for camera API access. |
| **API Server** | FastAPI, Uvicorn, Pydantic v2 | Endpoint orchestration, request validation, logging, and error handling. |
| **Quality Module** | OpenCV (`cv2`) | Fast assessment of motion blur, darkness, saturation, and reference patch detection. |
| **Colour & ROI Module** | OpenCV, NumPy | Colour constancy (Shades of Gray), background/spacer ring exclusion, and lesion bounding crop. |
| **Inference Engine** | ONNX Runtime (CPU) / PyTorch | High-throughput, low-latency evaluation of the trained EfficientNet-B0 network. |
| **Calibration Module** | NumPy, SciPy | Post-hoc temperature scaling ($T$) ensuring reliable, clinically calibrated probabilities. |
| **Triage Rules Engine** | Python | Maps probability vector to actionable clinical guidance (`ok`, `uncertain`, `urgent`). |
| **Explainability Engine**| PyTorch Grad-CAM | Generates morphological activation heatmaps confirming model attention. |

---

## 3. Data Flow (Single Mobile Capture)

1. **Client Capture:** User aligns the spacer flush against the skin. The video frame is snapped at native sensor resolution and encoded to JPEG (quality 0.92).
2. **Payload Submission:** `POST /api/v1/predict` sends:
   - `image`: Image binary (multipart/form-data)
   - `modality`: `"mobile_spacer"` (or `"dermatoscope"` in later phases)
   - `device_info`: Mobile handset model and browser agent
   - `capture_height_mm`: Distance from lens to skin (e.g. `35.0`)
3. **Quality Validation:** OpenCV computes Laplacian variance and mean brightness. If blurred or poorly lit, returns HTTP 422 with actionable guidance (e.g., `"Hold spacer steady against skin"`).
4. **Colour Normalization:** Calculates illumination cast using Gray-World or calibrated reference patch and standardizes color temperature across varied environments.
5. **Lesion Localization & ROI Cropping:** Identifies central lesion region, crops out the dark circular spacer borders, and resizes to $224 \times 224$ pixels.
6. **Inference & Calibration:** Evaluates ONNX model to produce raw logits $z_i$, then computes calibrated probabilities $p_i = \text{softmax}(z_i / T)$.
7. **Triage Assessment:**
   - If $P(\text{suspicious\_lesion}) \ge 0.25 \implies$ `urgent` triage alert.
   - If $\max_i P(i) < 0.50$ or $(P_1 - P_2) < 0.10 \implies$ `uncertain` triage alert.
   - Otherwise $\implies$ `ok` triage with condition name and reassurance advice.
8. **Client Display:** Renders triage alert banner, condition details, and top-3 distribution directly on the mobile screen.

---

## 4. Hosting & Network Topologies

Because **mobile browsers strictly forbid camera access on unencrypted HTTP connections** (except `localhost`, which a phone cannot access on an external machine), the hosting topology requires an HTTPS endpoint:

```
[Phone on 4G/Wi-Fi] 
         │
         ▼ (HTTPS: https://medha-demo.trycloudflare.com)
[Cloudflare Tunnel / ngrok Ingress]
         │
         ▼ (HTTP localhost:8000)
[FastAPI Backend on Dev Laptop]
```

| Deployment Mode | Ingress Strategy | Best Suited For |
|---|---|---|
| **Local Development with Phone** | **Cloudflare Tunnel (`cloudflared tunnel --url http://localhost:8000`)** or **ngrok** | Rapid live testing on physical smartphone with zero server cost. |
| **Local Wi-Fi Network** | Self-signed SSL certificate with local DNS or mkcert | Offline environments without active internet connection. |
| **Demo Day / Production** | Cloud Run or Render container with custom domain and managed SSL | High-reliability backup for hackathon jury evaluation. |

---

## 5. Configuration Architecture (`configs/classes.yaml`)

```yaml
modality: mobile_spacer
version: "v2.0-mobile"

classes:
  - id: eczema
    label: "Eczema"
    name: "Atopic Dermatitis / Eczema"
    severity: "common"
    color: "#2563EB"
  - id: psoriasis
    label: "Psoriasis"
    name: "Psoriasis"
    severity: "common"
    color: "#D97706"
  - id: tinea
    label: "Tinea"
    name: "Tinea / Fungal Ringworm"
    severity: "common"
    color: "#7C3AED"
  - id: acne
    label: "Acne"
    name: "Acne Vulgaris"
    severity: "common"
    color: "#059669"
  - id: healthy
    label: "Healthy Skin"
    name: "Normal Skin"
    severity: "normal"
    color: "#10B981"
  - id: suspicious_lesion
    label: "Suspicious Lesion"
    name: "Atypical / Neoplastic Lesion"
    severity: "urgent"
    color: "#DC2626"

thresholds:
  uncertain_max_prob: 0.50
  margin_min_prob: 0.10
  urgent_suspicious_prob: 0.25
  blur_min_laplacian_var: 55.0
  brightness_range: [40, 225]
  glare_max_ratio: 0.12
```

---

## 6. Security, Privacy & Data Protection

- **No EXIF Footprint:** Smartphone cameras embed exact GPS coordinates, timestamps, and device identifiers. The backend immediately strips all EXIF metadata in memory upon receiving the byte stream.
- **In-Memory Volatility:** Images are processed entirely within RAM buffers (`io.BytesIO`) and discarded upon request completion unless explicit user consent is flagged for model refinement.
- **Origin Isolation:** Strict CORS middleware allowing only the authorized frontend web origin.
