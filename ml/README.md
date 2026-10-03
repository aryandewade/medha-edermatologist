# 6-Class Skin Disease Classifier (EfficientNet-B0)

Mobile-spacer skin screening: EfficientNet-B0 trained on smartphone + 3D-printed
optical spacer captures, with reference-patch colour calibration and calibrated
clinical triage.

## Supported Classes (6)
1. **Eczema** (Atopic Dermatitis)
2. **Psoriasis**
3. **Tinea** (Ringworm / fungal infection)
4. **Acne**
5. **Healthy Skin**
6. **Suspicious Lesion** (neoplastic / pre-cancerous — urgent referral)

---

## Workflow Architecture

```
[Smartphone + 3D-Printed Spacer]
          ↓
   Mobile PWA Capture (HTTPS)
          ↓
   Image Preprocessing
┌──────────────────────────────────────┐
│ • Quality Check (Blur / Glare / Illum)│
│ • Optional ROI Crop (Center Square)  │
│ • Resize to 224 × 224                │
│ • ImageNet Normalize                 │
└──────────────────┬───────────────────┘
                   ↓
         EfficientNet-B0 Backbone
          (ImageNet Pretrained)
                   ↓
      Custom Classifier Head
      (Dropout 0.3 + Linear 1280→6)
                   ↓
       6 Calibrated Output Classes
 ┌──────────┬───────────┬────────┬────────┬────────────┬──────────────────────┐
 ↓          ↓           ↓        ↓        ↓            ↓                      ↓
Eczema  Psoriasis   Tinea    Acne   Healthy   Suspicious Lesion     [Triage / Abstain]
```

---

## File Structure

- [`model.py`](src/model.py): `SkinDiseaseEfficientNetB0` PyTorch architecture with custom 6-class head and feature extraction.
- [`preprocess.py`](src/preprocess.py): Quality gate (Laplacian blur, glare, brightness), reference-patch colour constancy + Shades-of-Gray fallback, ROI cropping, 224x224 normalization.
- [`reference_patch.py`](src/reference_patch.py): 18% gray reference-patch locator + white balance (adaptive neutrality vs. skin baseline).
- [`inference.py`](src/inference.py): End-to-end triage engine (quality -> colour -> localization -> calibrated inference -> referral rules).
- [`dataset.py`](src/dataset.py): Dataset loader with mobile-appropriate augmentations and inverse class weighting.
- [`train.py`](src/train.py): Two-stage transfer learning (frozen warmup + unfrozen fine-tuning with cosine decay).
- [`calibrate.py`](src/calibrate.py): Temperature scaling (NLL fit + ECE) -> writes `backend/models/temperature.json`.
- [`eval.py`](src/eval.py): Per-class Precision/Recall/Macro-F1, confusion matrix (CSV+PNG), Top-3 accuracy, ECE, Fitzpatrick subgroup audit.
- [`data_harmonize.py`](src/data_harmonize.py): Ingest SCIN/Fitzpatrick17k/PAD-UFES-20 -> 6 canonical classes, pHash dedupe, group-safe stratified split + manifest.
- [`export_onnx.py`](src/export_onnx.py): ONNX model exporter for high-speed CPU inference.
- [`classes.yaml`](../configs/classes.yaml): Configuration defining class labels, colors, and clinical uncertainty thresholds.

---

## Quick Usage

All commands run from the `ml/` directory.

### 1. Harmonize datasets into the 6 canonical classes
```bash
python src/data_harmonize.py \
  --fitzpatrick_dir raw/fitzpatrick17k \
  --scin_dir raw/scin \
  --pad_dir raw/pad-ufes-20 \
  --local_dir raw/spacer_captures \
  --out_dir data --dedupe
```

### 2. Train / Fine-tune
Organize (or let step 1 produce) the dataset as:
```
data/
  train/<class>/*.jpg
  val/<class>/*.jpg
  test/<class>/*.jpg
```
Then run:
```bash
python src/train.py --train_dir data/train --val_dir data/val --finetune_epochs 20
```

### 3. Calibrate (temperature scaling)
```bash
python src/calibrate.py --checkpoint models/best_model.pt --val_dir data/val
```
Writes `backend/models/temperature.json`; the inference engine loads it automatically.

### 4. Evaluate (metrics + confusion matrix + fairness)
```bash
python src/eval.py --checkpoint models/best_model.pt --test_dir data/test --output_dir reports
```

### 5. Export to ONNX (for fast CPU inference)
```bash
python src/export_onnx.py --checkpoint models/best_model.pt --output models/model.onnx
```

### Reference-patch calibration (standalone check)
```bash
python src/reference_patch.py --image sample_capture.jpg
```
