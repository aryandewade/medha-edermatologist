# Dual-Agent Simultaneous Development & Coordination Plan

**Project:** E-Dermatologist (Mobile Spacer Edition)  
**Coordination Strategy:** Strict modular separation with immutable interface contracts to enable simultaneous, conflict-free development.

---

## 1. Modular Division of Labor

```
┌─────────────────────────────────────────────────────────────┐
│                    SHARED CONTRACTS                         │
│  • configs/classes.yaml (Class Schema & Thresholds)         │
│  • api-spec.md          (REST API Data Contracts)           │
│  • models/model.onnx    (Shared Model Weights Artifact)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌──────────────────────────────┐┌──────────────────────────────┐
│  AGENT 1: ANTIGRAVITY        ││  AGENT 2: PARTNER AGENT      │
│  (Frontend, UX & Reporting)  ││  (ML, Vision & Hardware)     │
├──────────────────────────────┤├──────────────────────────────┤
│ • Mobile PWA UI & Viewfinder ││ • Data Harmonization (SCIN/  │
│ • i18n (EN, HI, MR strings)  ││   Fitzpatrick17k/PAD-UFES)   │
│ • PDF Referral Report Export ││ • Reference Patch CV Module  │
│ • Backend /report Endpoint   ││ • Temperature Calibration (T)│
│ • Client State & History     ││ • Evaluation Script & Matrix │
│ • Touch Ergonomics & Audio   ││ • 3D CAD Spacer STL Generator│
└──────────────────────────────┘└──────────────────────────────┘
```

---

## 2. Strict File Ownership Matrix

Neither agent should edit files owned by the other without explicit coordination.

| Subsystem / Directory | Owner | Permitted Actions |
|---|---|---|
| `frontend/src/**` | **Agent 1 (Antigravity)** | Read/Write (Components, styling, i18n, PDF export) |
| `backend/app/services/report.py` | **Agent 1 (Antigravity)** | Read/Write (ReportLab PDF generation service) |
| `ml/src/data_harmonize.py` | **Agent 2 (Partner)** | Read/Write (Dataset downloading and label mapping) |
| `ml/src/calibrate.py` | **Agent 2 (Partner)** | Read/Write (Temperature scaling optimization) |
| `ml/src/eval.py` | **Agent 2 (Partner)** | Read/Write (Confusion matrix, subgroup fairness) |
| `ml/src/reference_patch.py` | **Agent 2 (Partner)** | Read/Write (OpenCV reference patch locator) |
| `hardware/**` | **Agent 2 (Partner)** | Read/Write (3D print CAD, STL generators, scale bars) |
| `configs/classes.yaml` | **SHARED (Frozen)** | Read-Only unless mutually agreed in this document |
| `backend/app/routers/predict.py`| **SHARED (Careful)** | Agent 1 owns `/report`; Agent 2 integrates CV patch |

---

## 3. Simultaneous Task Schedule

### Agent 1 (Antigravity) Work Queue:
- [x] **Task 1.1:** Build `backend/app/services/report.py` using ReportLab to generate a clean, 1-page clinical referral PDF.
- [x] **Task 1.2:** Add `POST /api/v1/report` endpoint to `backend/app/routers/predict.py` returning binary PDF.
- [x] **Task 1.3:** Implement Hindi (`hi`) and Marathi (`mr`) full translation dictionaries in `frontend/src/i18n.ts`.
- [x] **Task 1.4:** Connect the **PDF Export** button in `ResultsDashboard.tsx` to download the generated report.
- [x] **Task 1.5:** Add local session history drawer (allowing health workers to review previous captures).

### Agent 2 (Partner Agent) Work Queue:
- [x] **Task 2.1:** Create `ml/src/data_harmonize.py` to ingest and map SCIN, Fitzpatrick17k, and PAD-UFES-20 into the 6 canonical classes.
- [x] **Task 2.2:** Build `ml/src/reference_patch.py` using OpenCV template/color matching to locate the 18% gray patch in the spacer margin.
- [x] **Task 2.3:** Build `ml/src/calibrate.py` to optimize temperature $T$ on validation logits and output `backend/models/temperature.json`.
- [x] **Task 2.4:** Build `ml/src/eval.py` computing per-class Precision, Recall, Macro-F1, and Confusion Matrix.
- [x] **Task 2.5:** Create `hardware/spacer_generator.py` (or OpenSCAD script) generating parameterized 35mm optical spacer STL models.

---

## 4. Live Status Tracker

| Task ID | Description | Assigned To | Status | Notes |
|---|---|---|---|---|
| **1.1** | ReportLab PDF referral generator service | Agent 1 (Antigravity) | `COMPLETED` | Generates 1-page PDF summary |
| **1.2** | `POST /api/v1/report` backend endpoint | Agent 1 (Antigravity) | `COMPLETED` | Endpoint verified with HTTP 200 |
| **1.3** | Hindi & Marathi i18n dictionaries | Agent 1 (Antigravity) | `COMPLETED` | Dynamic EN/HI/MR switcher active |
| **1.4** | PDF export button integration in React UI | Agent 1 (Antigravity) | `COMPLETED` | One-touch PDF download |
| **1.5** | Session history drawer in mobile UI | Agent 1 (Antigravity) | `COMPLETED` | Stores last 8 captures in session |
| **2.1** | `ml/src/data_harmonize.py` mapping pipeline | Agent 2 (Partner) | `COMPLETED` | 6-class label map, pHash dedupe, group-safe stratified split, manifest |
| **2.2** | `ml/src/reference_patch.py` locator | Agent 2 (Partner) | `COMPLETED` | Adaptive gray-patch WB; integrated into `preprocess.py` + `inference.py` + `/predict` |
| **2.3** | `ml/src/calibrate.py` temperature scaling | Agent 2 (Partner) | `COMPLETED` | ECE + scipy T-fit; writes `backend/models/temperature.json`; loaded by engine |
| **2.4** | `ml/src/eval.py` metrics and matrix | Agent 2 (Partner) | `COMPLETED` | Precision/Recall/Macro-F1, CM (CSV+PNG), Top-3, ECE, Fitzpatrick audit |
| **2.5** | 3D spacer parametric CAD script | Agent 2 (Partner) | `COMPLETED` | `hardware/spacer_generator.py` + `spacer.scad`; watertight STL in `hardware/stl/` |

---

## 5. Clean Interface Contracts for Agent 2

To start work immediately without conflicts, Agent 2 should follow these exact interfaces:

### Interface 1: Reference Patch Detector (`ml/src/reference_patch.py`)
```python
def detect_and_calibrate_patch(image_bgr: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Returns:
      corrected_bgr: Normalized image array
      patch_found: True if 18% gray patch was located on spacer margin
      metrics: {"gray_deviation": float, "patch_coords": [x, y, w, h]}
    """
```

### Interface 2: Temperature Calibration Output (`backend/models/temperature.json`)
```json
{
  "temperature": 1.25,
  "ece_before": 0.124,
  "ece_after": 0.038,
  "calibrated_at": "2026-10-03"
}
```

### Interface 3: Hardware CAD (`hardware/`)
Create `hardware/spacer_generator.py` or `.scad` to output STL files into `hardware/stl/`.
