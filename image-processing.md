# Digital Image Processing Pipeline (Mobile Spacer Photography)

This document details the complete computer vision and digital image processing pipeline executed prior to model inference. Because frontline smartphone photos suffer from varied illuminants, hand tremors, and skin reflections, this preprocessing sequence transforms noisy field captures into standardized, normalized clinical inputs.

---

## 1. High-Level Processing Architecture

```
Raw Smartphone Frame (JPEG, 12MP / 4032×3024)
                      │
                      ▼
        [Stage 1: Quality Gate & Validation]
        ├─ Laplacian Sharpness (Var ≥ 55.0)
        ├─ Illumination Range ([40, 225])
        ├─ Glare / Specular Hotspot Ratio (≤ 12%)
        └─ Reference Patch Detection
                      │ (PASS)
                      ▼
        [Stage 2: Colour Constancy & Normalization]
        ├─ Shades-of-Gray Illuminant Estimation (p=6)
        └─ Reference Patch Alignment (Gray R=G=B)
                      │
                      ▼
        [Stage 3: Lesion Localization & ROI Segmentation]
        ├─ Spacer Rim & Background Masking
        ├─ Skin Thresholding (YCbCr / HSV)
        ├─ Primary Lesion Bounding Box
        └─ Aspect-Ratio Preserving Square Crop
                      │
                      ▼
        [Stage 4: Tensor Standardization]
        ├─ Bicubic Resize to 224 × 224
        └─ ImageNet Normalization (Mean / Std)
                      │
                      ▼
        To EfficientNet-B0 Backbone (1, 3, 224, 224)
```

---

## 2. Stage 1: Quality Gates & Automated Validation

Images failing physical capture criteria are rejected immediately with actionable guidance before consuming ML inference cycles:

### A. Blur & Defocus Estimation (Laplacian Variance)
The Laplacian operator $\nabla^2 I$ computes the second-order spatial derivatives of pixel intensities:
$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
Sharp edges exhibit high spatial variance, whereas defocused or motion-blurred frames exhibit low variance:
$$\text{Blur Score} = \text{Var}\left( \nabla^2 I_{\text{gray}} \right) = \frac{1}{N} \sum_{x,y} \left( \nabla^2 I(x,y) - \mu_{\nabla^2} \right)^2$$
- **Threshold:** $\text{Blur Score} \ge 55.0$ (Pass). If $< 55.0$, return HTTP 422: `"Image too blurry. Hold spacer steady against skin."`

### B. Illumination & Dynamic Range
- Convert to grayscale $I_{\text{gray}}$. Calculate mean intensity $\mu_I$.
- **Acceptance Window:** $40 \le \mu_I \le 225$.
- Images $< 40$ are underexposed (insufficient clinic lighting); images $> 225$ are overexposed.

### C. Specular Glare & Reflection Ratio
Moist or oily skin reflects direct light rays, producing blinding saturated white spots ($I \ge 245$ across all channels):
$$\text{Glare Ratio} = \frac{\sum \mathbb{I}(I_{\text{gray}} \ge 245)}{N_{\text{total}}}$$
- **Threshold:** $\text{Glare Ratio} \le 0.12$ ($12\%$). If higher, the user is prompted to tilt the spacer slightly to dissipate direct specular bounce.

---

## 3. Stage 2: Colour Constancy & Illumination Normalization

Smartphone cameras apply proprietary Auto White Balance (AWB) algorithms that alter skin tones under fluorescent, halogen, or warm LED lighting. We apply physical colour constancy to extract the true skin reflectance:

### A. Shades-of-Gray Algorithm
Generalizes the Gray-World ($p=1$) and White-Patch ($p=\infty$) hypotheses using the Minkowski $p$-norm ($p=6$):
$$e_c = \left( \frac{1}{|A|} \int_{x \in A} |I_c(x)|^p dx \right)^{1/p}, \quad c \in \{R, G, B\}$$
The estimated illuminant vector $\mathbf{e} = [e_R, e_G, e_B]$ is normalized to unit length:
$$\hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|}$$
Pixel values are corrected to a neutral daylight illuminant $[1/\sqrt{3}, 1/\sqrt{3}, 1/\sqrt{3}]$:
$$I_c^{\text{corrected}}(x) = I_c(x) \cdot \frac{1}{\sqrt{3} \cdot \hat{e}_c}$$

### B. Reference-Patch White Balance (High-Precision Calibration)
When the calibrated 18% neutral gray patch on the spacer rim is detected:
1. Sample mean pixel intensities $[\bar{R}_{\text{ref}}, \bar{G}_{\text{ref}}, \bar{B}_{\text{ref}}]$ over the patch ROI.
2. Compute gain factors relative to green:
   $$k_R = \frac{\bar{G}_{\text{ref}}}{\bar{R}_{\text{ref}}}, \quad k_G = 1.0, \quad k_B = \frac{\bar{G}_{\text{ref}}}{\bar{B}_{\text{ref}}}$$
3. Scale every pixel: $R' = k_R \cdot R$, $B' = k_B \cdot B$.

---

## 4. Stage 3: Lesion Localization & ROI Segmentation

Digital spacers capture an internal circular field of view bordered by the matte-black spacer flange. We isolate the clinical lesion from the surrounding physical border:

### A. Spacer Ring & Background Masking
- The spacer contact ring creates a known circular aperture centered at $(x_c, y_c)$ with radius $R_{\text{spacer}} \approx 0.45 \cdot \min(H, W)$.
- A binary circular mask $M_{\text{circ}}(x,y)$ zeroes out all pixels outside the aperture, preventing dark plastic borders from confounding the classifier.

### B. Skin & Lesion Segmentation
- Convert the masked RGB image to the $YC_bC_r$ color space.
- Apply empirical skin chrominance constraints:
  $$77 \le C_b \le 127 \quad \text{and} \quad 133 \le C_r \le 173$$
- Calculate morphological Otsu thresholding within the skin region to segment the darker, erythematous, or hyperpigmented lesion area from normal background skin.
- Apply morphological closing (kernel size $5 \times 5$) to bridge small gaps caused by skin pores or fine hairs.

### C. Aspect-Ratio Preserving Square Bounding Crop
- Compute the bounding box $(x_1, y_1, w_{\text{box}}, h_{\text{box}})$ enclosing the largest connected lesion component.
- Expand to a square of side length $S = \max(w_{\text{box}}, h_{\text{box}}) \times 1.25$ (providing a $25\%$ surrounding healthy margin for context).
- If no discrete focal lesion is found (as in widespread eczema or normal healthy skin), default to the central square crop of the clear aperture.

---

## 5. Stage 4: Standardization for Deep Learning

1. **Spatial Resampling:**
   - Crop the segmented square region and downsample to $224 \times 224$ pixels using **Bicubic Interpolation** (`cv2.INTER_CUBIC`), preserving high-frequency textural patterns (e.g. scales, papules, pigment network).
2. **Dynamic Range Scaling:**
   - Convert `uint8` $[0, 255]$ to `float32` $[0.0, 1.0]$.
3. **Statistical Normalization:**
   - Normalize each color channel using ImageNet population statistics:
     $$x_{\text{norm}}^{(c)} = \frac{x^{(c)} - \mu_c}{\sigma_c}$$
     where $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ and $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$.
4. **Memory Layout:**
   - Permute channel ordering from OpenCV $(H, W, C)$ to PyTorch Tensor $(1, C, H, W)$.

---

## 6. Execution Latency Profile (Laptop CPU Benchmark)

| Stage | Operation | CPU Latency (Intel i5/i7) |
|---|---|---|
| **Quality Gate** | Grayscale conversion + cv2.Laplacian + pixel sum | $4.2\text{ ms}$ |
| **Colour Constancy** | Shades-of-Gray Minkowski norm ($p=6$) | $12.5\text{ ms}$ |
| **Lesion Segmentation**| Masking + $YC_bC_r$ thresholding + contour box | $8.1\text{ ms}$ |
| **Resampling & Tensor**| Bicubic resize to $224 \times 224$ + standardization | $2.4\text{ ms}$ |
| **Total Pipeline** | **End-to-End Image Processing** | **$\approx 27.2\text{ ms}$** |

The entire preprocessing sequence consumes under **$30\text{ ms}$** per frame on a standard laptop CPU, ensuring instant response times for mobile health workers.
