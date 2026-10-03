# Data Plan (Mobile Spacer-First)

## 1. Class Set (Mobile Clinical Profile)

The classification schema focuses on common outpatient conditions treatable at primary health centers, plus a healthy skin baseline and a critical referral category for suspicious neoplastic lesions.

| ID | Condition / Category | Primary Clinical Sources | Triage Action |
|----|----------------------|--------------------------|---------------|
| `eczema` | Eczema (Atopic Dermatitis) | DermNet, Fitzpatrick17k, SCIN, SD-198 | Primary care treatment / Monitor |
| `psoriasis` | Psoriasis | DermNet, Fitzpatrick17k, SCIN, SD-198 | Primary care treatment / Specialist triage |
| `tinea` | Tinea (Ringworm / Fungal) | DermNet, Fitzpatrick17k, SCIN | Antifungal treatment / Monitor |
| `acne` | Acne Vulgaris | DermNet, Fitzpatrick17k, SCIN, SD-198 | Primary care treatment |
| `healthy` | Healthy Skin | Own spacer captures, normal skin regions | Reassurance |
| `suspicious_lesion` | Suspicious Lesion (Melanoma, BCC, SCC, Actinic Keratosis) | DDI, PAD-UFES-20, Fitzpatrick17k, SCIN | **Urgent Dermatologist Referral** |

*Note on `suspicious_lesion`:* Rather than forcing a mobile phone camera to reliably sub-classify subtle histological differences between early melanoma, nodular BCC, and invasive SCC without dermoscopy, these high-risk neoplastic and pre-malignant lesions are pooled into a single high-sensitivity referral class. A specialized dermatoscope modality can be added later for deep lesion subtyping.

---

## 2. Source Datasets (Clinical & Smartphone Photography)

Dermoscopy datasets (HAM10000, ISIC) are replaced with clinical and smartphone macro photography datasets reflecting realistic mobile optical characteristics:

| Dataset | Modality & Description | Key Assets | Considerations & Caveats |
|---------|------------------------|------------|--------------------------|
| **Fitzpatrick17k** | Clinical & patient-captured skin photos | 16,577 images labeled with condition and Fitzpatrick skin type (FST I–VI) | Atlases web-scraped; labels have minor noise; deduplicate carefully |
| **SCIN (Google)** | Smartphone photos crowdsourced in real-world settings | Real mobile sensors, diverse lighting, includes healthy/mild cases | Free-form framing; requires lesion cropping and quality filtering |
| **PAD-UFES-20** | Smartphone cameras (various models) | 2,298 clinical images with biopsy labels and patient metadata | Includes smartphone photos of suspicious lesions and inflammatory diseases |
| **DDI (Diverse Dermatology Images)** | Clinical photography from Stanford | High-quality clinical photos across Fitzpatrick I–VI with biopsy ground truth | Excellent benchmark for fairness and suspicious lesion sensitivity |
| **DermNet NZ** | Clinical photography atlas | Extensive visual coverage of inflammatory and fungal conditions | Check licensing; resolution varies; standard clinical perspective |
| **SD-198** | Clinical digital photography | 198 disease classes, 6,585 images | Good supplemental source for acne, eczema, and psoriasis variations |
| **Own Spacer Captures** | Smartphone + 3D-printed spacer + reference patch | Exact hardware domain target for calibration and validation | Primary benchmark for model selection, thresholding, and demo stability |

---

## 3. Label Harmonization & Mapping

1. **Master Dictionary (`ml/data/label_map.csv`):** Maps thousands of raw source tags into the 6 canonical target IDs (`eczema`, `psoriasis`, `tinea`, `acne`, `healthy`, `suspicious_lesion`).
2. **Suspicious Lesion Grouping:**
   - Includes: Melanoma, Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Actinic Keratosis (AK), Atypical/Dysplastic Nevi with malignant suspicion.
   - Excludes: Benign regular nevi, seborrheic keratosis, and common freckles (filtered or routed to healthy/benign where verified).
3. **Ambiguity Handling:** Any image with conflicting multi-disease tags or low diagnostic confidence is discarded.

---

## 4. Data Cleaning & Normalization Pipeline

1. **Perceptual Deduplication:** Run perceptual hashing (pHash and dHash, Hamming distance $\le 4$) across all merged datasets to eliminate duplicates and near-identical crops that could bridge train and test sets.
2. **Artifact & Non-Skin Rejection:** Automatically detect and discard images with non-skin subjects, watermarks, intrusive text, or black borders.
3. **Lesion Localization & Standardized Cropping:**
   - Detect skin ROI and center crop/crop around lesion bounding boxes.
   - Standardize aspect ratios to prevent unnatural stretching.
4. **Colour & Illumination Normalization:**
   - Apply Gray-World or Shades-of-Gray colour constancy to correct varied hospital/room lighting casts before training.

---

## 5. Dataset Splits

| Split | Proportion | Splitting Strategy |
|---|---|---|
| **Train** | 70% | Stratified by class and grouped by patient/source ID to prevent data leakage. |
| **Validation** | 15% | Used for hyperparameter tuning, early stopping, and temperature scaling calibration. |
| **Public Test** | 15% | Held out for benchmark evaluation. |
| **Real Spacer Test Set** | Separate (Held-out) | Real photos captured with the project smartphone and 3D spacer. Serves as the ultimate ground-truth decision gate. |

---

## 6. Real-Camera Spacer Data Collection Protocol

To eliminate domain shift and calibrate the mobile model on the physical hardware:

### Hardware Setup
- **Smartphone:** Standardized test model (e.g., primary camera, 1x focal length, autofocus locked/set to spacer height, flash disabled, ambient room lighting or internal spacer diffuse LED).
- **3D-Printed Spacer:** Fixed height (e.g., 35 mm distance from camera lens to skin plane), matte-black non-reflective interior.
- **Reference Patch:** A calibrated dual-color/gray reference ring (18% neutral gray, 90% white patch, and 5 mm scale bar) mounted at the spacer contact perimeter to calibrate scale and white balance.

### Collection Procedure
1. **Healthy Skin Targets:**
   - Minimum 100 captures across 10+ diverse individuals covering Fitzpatrick types III to VI.
   - Body sites: Inner forearm, outer arm, dorsal hand, cheek, forehead, neck.
   - 3 captures per site (repositioned between shots).
2. **Clinical Lesions (Hospital Collaboration):**
   - Conducted exclusively under clinician supervision at Kaushalya Hospital.
   - **Mandatory written informed consent** signed before capturing.
   - Record anonymous metadata: Unique ID, age band, sex, anatomical site, clinician ground truth, Fitzpatrick skin type, and reference patch visibility.
3. **Sanitization:** Disinfect the spacer contact ring with hospital-grade 70% isopropyl alcohol wipes between every patient.
4. **Data Hygiene:** Strip all EXIF metadata, GPS coordinates, device serial numbers, and timestamp identifiers before ingest into the training/eval folders.

---

## 7. Class Balance Targets & Bias Auditing

- **Training Distribution Target:** Aim for at least 800 training samples per class; floor of 300 samples for the `suspicious_lesion` cluster.
- **Class Imbalance Strategy:** Use inverse frequency class weighting and Focal Loss ($\gamma = 2.0$) to avoid bias toward frequent inflammatory conditions.
- **Fairness & Subgroup Audits:**
  - Evaluate accuracy, recall, and calibration error across Fitzpatrick skin type buckets: Fair (I–II), Intermediate (III–IV), and Deeply Pigmented (V–VI).
  - Safety requirement: Recall for `suspicious_lesion` must remain $\ge 0.90$ across all available skin-tone cohorts.
