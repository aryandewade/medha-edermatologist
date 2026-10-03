# Testing and Evaluation Protocol (Mobile Spacer)

## 1. Test Layers

| Layer | Target System | Methodology & Tools |
|---|---|---|
| **Unit (Backend)** | Preprocessing, blur metrics, colour constancy, calibration | `pytest`, synthetic test image matrices |
| **Unit (Mobile PWA)** | Camera stream, spacer framing reticle, one-touch capture | `Vitest`, React Testing Library, mock camera streams |
| **API Contract** | Multipart upload, schema compliance, HTTP error codes | `pytest`, `httpx` test client |
| **Model Benchmark** | Performance across Fitzpatrick skin types and test sets | `ml/src/eval.py`, scikit-learn, confusion matrices |
| **Real Hardware Gate**| End-to-end evaluation on physical phone + 3D spacer captures | Held-out real spacer test set (100+ images) |

---

## 2. Model Evaluation Metrics & Acceptance Criteria

| Metric | Target Criterion | Clinical & Operational Rationale |
|---|---|---|
| **Macro-F1 (6 classes)** | $\ge 0.78$ | Balanced performance across common and rare classes. |
| **Recall for `suspicious_lesion`** | **$\ge 0.90$** | **Highest Safety Priority:** False negatives on potential malignancies must be minimized. |
| **Top-3 Accuracy** | $\ge 0.92$ | Clinical utility in differentiating look-alike conditions (e.g. Eczema vs Psoriasis). |
| **Calibration (ECE)** | $\le 0.08$ | Ensures confidence percentages match empirical probability. |
| **End-to-End Latency** | $\le 1.8\text{ s}$ | Smooth user experience over 4G mobile networks. |
| **Demo Stability** | 10 of 10 consecutive captures succeed | Zero app crashes or camera disconnects during demonstration. |

---

## 3. Subgroup & Fairness Audit

Performance must not degrade across varied skin tones or anatomical regions:
- **Fitzpatrick Subgroups:** Evaluate per-class precision and recall separately for:
  - FST I–II (Fair / Light)
  - FST III–IV (Medium / Olive)
  - FST V–VI (Deep / Dark)
- **Acceptance Gate:** Recall for `suspicious_lesion` must remain $\ge 0.88$ across every evaluated Fitzpatrick cohort.
- **Anatomical Slices:** Assess consistency across forearm, dorsal hand, torso, and face.

---

## 4. Robustness Stress Tests

Evaluate accuracy degradation under real-world mobile imperfections:
1. **Defocus Blur:** Apply Gaussian blur ($\sigma \in [1.0, 3.0]$) to simulate spacer displacement.
2. **Colour Temperature Casts:** Shift illuminant white point from $3000\text{ K}$ (warm tungsten) to $6500\text{ K}$ (cool daylight); confirm colour constancy algorithm maintains predictions within $5\%$ probability delta.
3. **Sensor Noise:** Add Gaussian and salt-and-pepper noise corresponding to ISO 800+ mobile shots.

---

## 5. Demo-Day Physical Hardware Checklist

Prior to entering the demonstration room:

### Physical Hardware
- [ ] Primary smartphone charged to $> 80\%$; screen brightness locked to $75\%$.
- [ ] 3D-printed optical spacer inspected:
  - Phone mount snaps firmly without slipping.
  - Interior matte-black finish is clean with no glossy reflection spots.
  - Contact rim is smooth and crack-free.
- [ ] Calibrated reference patch is firmly affixed and clean.
- [ ] Hygiene kit: Pack of hospital-grade 70% isopropyl alcohol wipes.
- [ ] Backup smartphone with identical cradle mount available.

### Software & Connectivity
- [ ] Backend server running locally on laptop (`uvicorn app.main:app`).
- [ ] Model warmed up with dummy inference call (`/health` reports `warmed_up: true`).
- [ ] Cloudflare Tunnel active (`cloudflared tunnel --url http://localhost:5173`); HTTPS link loaded and bookmarked on the phone.
- [ ] Camera permissions granted in mobile browser.
- [ ] Completed 3 trial captures on team member skin.
- [ ] Emergency fallback: Offline sample gallery button loaded in the web app.
