# Product Requirements Document (PRD)

**Product:** E-Dermatologist (Mobile Spacer Edition)  
**Context:** Hackathon problem statement, Somaiya Vidyavihar University with Kaushalya Hospital  
**Status:** Updated Specification v2 (Mobile-First Architecture)

---

## 1. Background
In rural and semi-urban India, access to qualified dermatologists is severely restricted. Frontline health workers (e.g., ASHA workers, community clinic nurses) encounter hundreds of patients with unmanaged or misidentified skin problems. Providing a mobile-based screening tool that pairs everyday smartphones with an inexpensive **3D-printed optical spacer** enables standardized, close-up skin screening without requiring expensive specialized hardware at initial deployment.

---

## 2. Problem Statement
Build an accessible, mobile-first web system that captures standardized skin images using a smartphone fitted with a 3D-printed focal spacer. The system automatically verifies capture quality, normalizes lighting and colour, and classifies the presentation across common skin diseases, healthy skin, or suspicious lesions requiring urgent dermatologist referral.

---

## 3. Goals
1. **Zero-App-Store Friction:** Runs as a responsive Progressive Web App (PWA) accessed over HTTPS on standard mobile browsers (Chrome / Safari).
2. **Standardized Mobile Ingestion:** Leverages a 3D-printed spacer and optional reference patch to fix lens-to-skin distance, eliminate hand tremor focus shifts, and calibrate white balance.
3. **Clinical Triage Classification:** Multi-class classification covering common inflammatory, fungal, acneic, and healthy presentations, plus a high-sensitivity **Suspicious Lesion** triage gate.
4. **Transparent Clinical Uncertainty:** Abstains and flags low-confidence or unclassifiable presentations for direct specialist review rather than guessing.
5. **Fast Mobile-Optimized Inference:** Complete analysis returned to the handset within 2 seconds.

---

## 4. Non-Goals (Hackathon Scope)
- Standalone definitive clinical diagnosis or prescribing prescription drugs.
- Replacing histopathological biopsy or in-person dermatoscope dermatological evaluation.
- Storing personally identifiable patient data or long-term Electronic Medical Records (EMR).
- Classifying ultra-rare dermatological conditions outside the supported taxonomy.

---

## 5. Users & Personas

| Persona | Environment & Role | Key Workflow Needs |
|---------|--------------------|--------------------|
| **Community Health Worker (ASHA / Nurse)** | Primary user in field clinics or outreach camps using a smartphone. | One-thumb interface, clear visual positioning cues, zero medical jargon, definitive referral instructions. |
| **Consulting Dermatologist** | Secondary user receiving triaged referrals at Kaushalya Hospital. | Standardized focal distance, colour-calibrated imagery, class probability distribution, quality indicators. |
| **Hackathon Evaluator / Judge** | Evaluating clinical utility, engineering rigor, and deployment feasibility. | Live working demonstration on physical phone hardware with 3D spacer, verifiable metrics, safety guardrails. |

---

## 6. Core User Flow (Mobile Spacer Workflow)

```
[Attach 3D Spacer to Phone]
            ↓
[Open Mobile Web App (HTTPS)] ──> [Automatic Rear Camera Stream Active]
            ↓
[Place Spacer Flush on Skin] ──> [Live Screen: Framing Guide & Lighting Check]
            ↓
[Tap "Capture & Analyze"] ──> [Backend Preprocessing & Inference]
            ↓
[Review Triage Result]
  ├─ Healthy / Reassurance
  ├─ Common Condition (Eczema / Psoriasis / Tinea / Acne)
  ├─ Uncertain (Adjust lighting / re-take / consult doctor)
  └─ Urgent Referral (Suspicious Lesion Alert)
            ↓
[Optional: Export PDF Case Summary for Referral]
```

---

## 7. Supported Conditions (Mobile Clinical Set)

1. **Eczema (Atopic Dermatitis):** Pruritic, erythematous, scaly patches.
2. **Psoriasis:** Well-demarcated erythematous plaques with silvery scaling.
3. **Tinea (Ringworm):** Annular scaly fungal lesions with active erythematous borders.
4. **Acne Vulgaris:** Comedones, papules, pustules, and inflammatory nodules.
5. **Healthy Skin:** Normal physiological skin surface without active disease.
6. **Suspicious Lesion:** Neoplastic or pre-cancerous lesions (Melanoma, BCC, SCC, Actinic Keratosis) triggering immediate specialist referral.

*(Future Phase: Specialized Dermatoscope Modality for fine-grained lesion subtyping).*

---

## 8. Functional Requirements

| ID | Feature | Description | Priority |
|----|---------|-------------|----------|
| **F1** | Camera Detection & Environment Selection | Automatically selects rear smartphone camera (`facingMode: environment`) with macro/focus lock. | Must |
| **F2** | Live Spacer Framing Guide | Displays a circular overlay and target reticle matching the 3D spacer internal field of view. | Must |
| **F3** | Connectivity & System Check | One-tap test verifying camera feed, HTTPS handshake, and backend API readiness. | Must |
| **F4** | High-Res One-Touch Capture | Freezes the frame, captures full-resolution JPEG, and uploads to `/api/v1/predict`. | Must |
| **F5** | Quality & Illumination Validation | Server-side validation of blur (Laplacian variance), brightness, and reference patch detection. | Must |
| **F6** | Multi-Class Probability Display | Clear condition card, confidence score, and top-3 probability breakdown. | Must |
| **F7** | Suspicious Lesion Referral Banner | Prominent high-contrast red warning when suspicious lesion probability exceeds safety threshold ($\ge 0.25$). | Must |
| **F8** | Low-Confidence / Abstention State | Amber warning advising physical doctor review when prediction uncertainty is high ($\max P < 0.50$). | Must |
| **F9** | Colour Constancy & Calibration | Automatic Gray-World or reference-patch colour correction before model evaluation. | Should |
| **F10** | Heatmap Lesion Overlay | Optional Grad-CAM++ visualization highlighting morphological regions influencing inference. | Should |
| **F11** | Local Referral PDF Export | Clean 1-page summary including normalized photo, top findings, and clinician disclaimer. | Should |
| **F12** | Image Upload Fallback | Allows uploading a pre-captured photo from phone gallery when live camera cannot be accessed. | Should |
| **F13** | Multilingual UI | English, Hindi, and Marathi localized interfaces. | Could |
| **F14** | Dermatoscope Modality Toggle | Future-proof selector for switching between mobile spacer mode and USB dermatoscope mode. | Could |

---

## 9. Non-Functional Requirements

- **Performance:** Complete round-trip inference in under **2.0 seconds** over 4G/Wi-Fi connection.
- **Mobile Ergonomics:** Single-column layout operable with one thumb; touch target sizes $\ge 48 \times 48\text{ px}$.
- **Security & Privacy:** Mandatory HTTPS; zero persistent storage of patient photographs by default; all EXIF and GPS tags stripped prior to analysis.
- **Portability:** Cross-browser support across mobile Safari (iOS 15+) and mobile Google Chrome (Android 10+).

---

## 10. Success Metrics (Target Acceptance Criteria)

- **Model Performance (on real spacer test images):**
  - Macro-F1 across the 6 classes $\ge 0.78$.
  - Recall for `suspicious_lesion` $\ge 0.90$ (safety priority: false negatives must be minimized).
  - Top-3 accuracy $\ge 0.92$.
  - Expected Calibration Error (ECE) $\le 0.08$.
- **Operational Metrics:**
  - 10 consecutive field captures completed without crash or timeout.
  - First-time frontline worker completes capture flow without training in $< 60$ seconds.
