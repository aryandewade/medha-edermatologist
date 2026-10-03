# ML Methodology (Mobile Clinical Profile)

## 1. Task Definition
Automated multi-class screening of macro skin images captured via smartphone cameras fitted with a 3D-printed optical spacer. The model outputs calibrated probabilities across 6 classes:
1. `eczema` (Atopic Dermatitis)
2. `psoriasis` (Psoriasis)
3. `tinea` (Ringworm / Dermatophytosis)
4. `acne` (Acne Vulgaris)
5. `healthy` (Normal Healthy Skin)
6. `suspicious_lesion` (Neoplastic / Pre-cancerous / Atypical Lesions)

---

## 2. Approach: Transfer Learning on EfficientNet-B0

- **Backbone Selection:** `EfficientNet-B0` pretrained on ImageNet-1k. Chosen for exceptional parameter efficiency (5.3M parameters), rapid CPU inference latency ($< 35\text{ ms}$ on laptop/cloud CPU), and strong macro-feature extraction.
- **Classification Head:** 
  $$\text{Embedding (1280)} \longrightarrow \text{Dropout}(p=0.3) \longrightarrow \text{Linear}(1280, 6) \longrightarrow \text{Logits}$$
- **Training Strategy:**
  - **Phase 1 (Warmup):** Freeze backbone feature layers; train custom classifier head for 3 epochs with AdamW ($\text{lr} = 3 \times 10^{-4}$).
  - **Phase 2 (End-to-End Fine-Tuning):** Unfreeze entire backbone for 20–25 epochs using differential learning rates ($\text{lr}_{\text{backbone}} = 5 \times 10^{-5}$, $\text{lr}_{\text{head}} = 2 \times 10^{-4}$), Cosine Annealing learning rate schedule, and Focal Loss ($\gamma = 2.0$) with class weighting.

---

## 3. Preprocessing & Colour Normalization Pipeline

Mobile smartphone cameras exhibit wide variations in auto-white balance, sensor ISP color tuning, and ambient lighting. Images pass through a structured preprocessing sequence:

```
Raw Mobile Capture
       ↓
Quality Gate (Laplacian blur score ≥ 55, Brightness in [40, 225], Glare ≤ 12%)
       ↓
Colour Constancy (Shades-of-Gray / Reference-Patch White Balance)
       ↓
Lesion Localization & ROI Extraction (Crop out spacer ring & peripheral shadows)
       ↓
Resize to 224 × 224 (Bicubic Interpolation)
       ↓
Standardization (ImageNet Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225])
```

1. **Colour Constancy:**
   - Evaluates the illuminant colour using the **Shades-of-Gray** algorithm (Minkowski $p$-norm $p=6$):
     $$e(x) = \left( \frac{\int |I(x)|^p dx}{\int dx} \right)^{1/p}$$
   - When a calibrated reference patch is visible within the spacer margin, normalizes RGB channels to align the reference patch with neutral gray ($R=G=B$).
2. **Lesion Localization:**
   - Detects skin foreground against the spacer's matte-black contact boundary using Otsu thresholding in $YC_bC_r$ color space.
   - Extracts the bounding box enclosing the primary lesion and applies a symmetric square crop.

---

## 4. Mobile Domain Data Augmentations

Dermatoscope-specific augmentations (circular vignettes, black contact gel bubbles, simulated dermatoscopy hairs) are removed. In their place, we apply augmentations modeling real smartphone camera dynamics using Albumentations:

| Augmentation Type | Operators | Clinical / Physical Rationale |
|---|---|---|
| **Illumination & White Balance** | `ColorJitter`, `RandomToneCurve`, `RGBShift` | Simulates variations in room lighting (fluorescent, warm LED, daylight). |
| **Shadows & Occlusion** | `RandomShadow`, `CoarseDropout` | Models shadows cast by the smartphone body or health worker's hand. |
| **Optics & Motion** | `GaussianBlur`, `DefocusBlur`, `MotionBlur` | Simulates hand instability or slight defocus during phone positioning. |
| **Geometry** | `RandomRotate90`, `Rotate(degrees=180)`, `Affine(scale=(0.85, 1.15))` | Skin lesions have no canonical orientation; accounts for varying phone angles. |
| **Sensor Noise** | `GaussNoise`, `ImageCompression(quality_range=(75, 95))` | Simulates mobile CMOS sensor grain in dim clinic rooms and JPEG compression. |

---

## 5. Calibration & Temperature Scaling

Raw deep learning models are notoriously overconfident. We calibrate output probabilities via post-hoc **Temperature Scaling**:

1. Logits $z = [z_1, \dots, z_6]$ are scaled by a learned scalar parameter $T > 0$:
   $$P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
2. $T$ is optimized on the held-out validation set to minimize Negative Log-Likelihood (NLL).
3. Evaluated using the **Expected Calibration Error (ECE)** with 10 confidence bins:
   $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right| \le 0.08$$

---

## 6. Clinical Triage & Decision Logic

To ensure patient safety and practical field utility, the system adheres to strict clinical referral rules:

```
                ┌───────────────────────────────────┐
                │   Calibrated Probabilities P      │
                └─────────────────┬─────────────────┘
                                  │
                 Is P(suspicious_lesion) ≥ 0.25 ?
                                 / \
                                /   \
                              YES    NO
                              /       \
                             ▼         ▼
                 [URGENT REFERRAL]    Is max(P) < 0.50 OR (P₁ - P₂) < 0.10 ?
                 "Suspicious lesion              / \
                  flagged. Prompt               /   \
                  specialist review           YES    NO
                  recommended."               /       \
                                             ▼         ▼
                                    [UNCERTAIN STATE]  [CONFIDENT RESULT]
                                    "Uncertain.        "Possible [Condition].
                                     Consult doctor."   Primary care triage."
```

- **`urgent` Triage Rule:** If $P(\text{suspicious\_lesion}) \ge 0.25$, the application triggers a high-visibility urgent referral notification regardless of whether another condition has higher probability.
- **`uncertain` Abstention Rule:** If highest probability is below $0.50$, or the top two probabilities differ by less than $0.10$ (indicative of look-alike confusion between eczema and psoriasis), the system abstains from declaring a condition and advises manual review.

---

## 7. Explainability (Grad-CAM++)

Grad-CAM++ generates heatmaps on the final convolutional stage (`features[-1]`). This confirms whether the prediction is driven by true biological morphology (e.g., erythematous scaling, follicular comedones, pigment asymmetry) rather than peripheral skin folds, hair, or shadows from the spacer wall.
