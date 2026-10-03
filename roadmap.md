# Roadmap and Implementation Plan (Mobile-First)

The implementation sequence prioritizes delivering a reliable mobile smartphone screening experience with a 3D-printed spacer, moving specialized dermatoscope hardware to a subsequent multimodal phase.

---

## Team Roles & Ownership

| Role | Core Focus | Primary Deliverables |
|---|---|---|
| **Hardware & Optics Lead** | Smartphone spacer design, 3D printing, reference patch | 3D CAD model, printed spacers, physical calibration |
| **ML & Image Processing Lead** | Data harmonization, model training, colour constancy, calibration | EfficientNet-B0 weights, preprocessing module, ONNX export |
| **Mobile & Backend Lead** | Mobile PWA (React), FastAPI backend, HTTPS ingress | PWA camera interface, REST endpoints, tunnel setup |
| **Clinical QA & Protocol Lead**| Data collection protocol, hospital testing, consent, ethics | Real-camera dataset, model card, referral evaluation |

---

## Phase 0: Hardware Selection & Environmental Setup
- [ ] Select primary smartphone reference model (e.g. Samsung Galaxy / OnePlus / iPhone) and document focal length and lens offset.
- [ ] Design 3D CAD model for the optical spacer cone (fixed height, matte interior, phone mounting clip).
- [ ] 3D print prototype spacer using biocompatible or skin-safe filament (black PLA/PETG).
- [ ] Mount or print a calibrated 18% neutral gray reference patch and 5 mm scale bar onto the spacer perimeter.
- [ ] Set up HTTPS ingress (Cloudflare Tunnel or ngrok) to enable phone camera access to the local development environment.

---

## Phase 1: Mobile Walking Skeleton (PWA + API)
- [ ] Implement responsive mobile-first frontend in React + Vite:
  - Rear camera video stream (`facingMode: environment`).
  - Circular spacer framing guide and central crosshair reticle.
  - Large touch-friendly **Capture & Analyze** button.
- [ ] Implement FastAPI backend endpoints (`/api/v1/health`, `/api/v1/classes`, `/api/v1/predict` with mock output).
- [ ] Verify end-to-end capture on the physical phone over HTTPS in $< 1.5$ seconds.

---

## Phase 2: Mobile Dataset & Baseline Model
- [ ] Download clinical datasets: Fitzpatrick17k, SCIN, PAD-UFES-20, DDI, DermNet.
- [ ] Execute label mapping into the 6 canonical classes (`eczema`, `psoriasis`, `tinea`, `acne`, `healthy`, `suspicious_lesion`).
- [ ] Run perceptual hash deduplication to eliminate cross-split duplicates.
- [ ] Train baseline EfficientNet-B0 with ImageNet transfer learning.
- [ ] Collect first 50 real spacer captures of healthy skin from team members.

---

## Phase 3: Image Processing & Model Fine-Tuning
- [ ] Implement the image processing pipeline:
  - Real-time Laplacian blur quality filter.
  - Shades-of-Gray / reference patch colour constancy.
  - Lesion localization and automatic square ROI cropping.
- [ ] Execute two-stage fine-tuning with mobile augmentations (shadows, white-balance shifts, defocus).
- [ ] Perform post-hoc Temperature Scaling to minimize calibration error (ECE $\le 0.08$).
- [ ] Verify high recall for `suspicious_lesion` ($\ge 0.90$).
- [ ] Export optimized model to ONNX for CPU serving.

---

## Phase 4: Integration, Field Hardening & Triage Rules
- [ ] Connect trained ONNX model and colour pipeline to the FastAPI `/predict` router.
- [ ] Implement clinical triage rule engine:
  - `urgent` referral banner for suspicious lesions ($P \ge 0.25$).
  - `uncertain` abstention banner for low-confidence or look-alike cases.
  - `ok` common condition summary card.
- [ ] Implement optional Grad-CAM++ morphological heatmap toggle.
- [ ] Conduct sanitization and repeatability tests with hospital-grade alcohol wipes on the 3D spacer.

---

## Phase 5: Demo Day Readiness & Multimodal Extension
- [ ] Execute end-to-end demo checklist on physical phone and spacer.
- [ ] Prepare offline sample image gallery as fallback.
- [ ] Generate 1-page PDF referral summary card.
- [ ] **Extension (Multimodal):** Add USB dermatoscope toggle (`modality: "dermatoscope"`) for high-magnification contact imaging.

---

## Milestone Milestones

| Milestone | Deliverable | Success Criteria |
|---|---|---|
| **M1: Mobile Skeleton** | Phone connects over HTTPS, streams camera, captures frame, displays mock results. | Latency $< 1.5\text{ s}$ on phone. |
| **M2: Hardware & Baseline** | 3D spacer printed; phone snaps flush skin photos; baseline 6-class model running. | Focus sharpness verified. |
| **M3: Full Pipeline** | Preprocessing, colour constancy, lesion localization, fine-tuned EfficientNet-B0. | Macro-F1 $\ge 0.78$ on test set. |
| **M4: Demo-Ready** | Real spacer test set evaluated, triage banners validated, 10/10 captures succeed. | Suspicious recall $\ge 0.90$. |
| **M5: Presentation** | Slide deck, live phone demo, physical 3D spacer showcase, model card. | Jury presentation rehearsed. |
