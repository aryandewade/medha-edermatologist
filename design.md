# Mobile UI/UX Design (Smartphone + 3D Spacer)

## 1. Design Principles
1. **One-Thumb Operation:** Primary actions (capture, retry, review) positioned in the natural thumb zone at the bottom of the screen.
2. **Physical-Digital Alignment:** Clear visual overlays guide the health worker to position the 3D spacer flush against the patient's skin.
3. **Clinical Triage Clarity:** Results emphasize triage action (Self-care/Monitor, Specialist Review, Urgent Referral) rather than raw disease probabilities.
4. **Resilient to Field Conditions:** Clear, actionable feedback on poor focus, inadequate light, or tilted spacer placement.

---

## 2. Mobile Screen Layout (Single-Column Vertical Flow)

```
+---------------------------------------+
|  [Logo] E-Dermatologist   [EN/HI/MR]  |
+---------------------------------------+
|                                       |
|  [Live Camera Viewfinder]             |
|   ┌─────────────────────────────┐     |
|   │       (  ( O )  )           │     |
|   │   Circular Spacer Reticle   │     |
|   │   & Lesion Framing Ring     │     |
|   │                             │     |
|   │ [• Ref Patch OK] [• Light OK]│    |
|   └─────────────────────────────┘     |
|                                       |
|  [Instructions Card]                  |
|  "Place spacer flat against skin.     |
|   Center the lesion inside circle."   |
|                                       |
|  [ CAPTURE & ANALYZE ] (Large Button) |
|  [ ⚡ Test Link ]   [ 📁 Upload Photo]|
+---------------------------------------+
|                                       |
|  [Triage Referral Banner]             |
|  🚨 URGENT REFERRAL RECOMMENDED       |
|                                       |
|  [Condition Summary Card]             |
|  Suspicious Lesion (78.4% Confidence) |
|                                       |
|  [Differential Distribution (Top-3)]  |
|  ■ Suspicious Lesion  78.4%           |
|  ■ Actinic / Psoriasis 14.1%          |
|  ■ Normal / Healthy    7.5%           |
|                                       |
|  [Analyzed Image with Heatmap Toggle] |
|                                       |
|  [ 📄 Export Referral Summary PDF ]   |
+---------------------------------------+
|  Footer: "Screening tool only. Not a  |
|  medical diagnosis. Consult doctor."  |
+---------------------------------------+
```

---

## 3. Core Mobile Components

| Component | Visual Presentation & Behavior |
|---|---|
| **Viewfinder & Spacer Reticle** | Full-width 4:3 or 1:1 camera feed showing a circular translucent framing ring matching the inner diameter of the 3D spacer cone. |
| **Placement & Lighting Chips** | Real-time visual status pills floating over the live feed: <br>• **Focus:** Green (`Sharp`) vs Amber (`Defocused - Press flush`) <br>• **Light:** Green (`Illumination OK`) vs Red (`Too dark`) <br>• **Reference Patch:** Green (`Patch Detected`) |
| **Primary Capture CTA** | Prominent floating action bar at bottom screen ($56\text{ px}$ height) with haptic feedback vibration on capture. |
| **Camera Selector (Streamlined)** | Automatically selects rear camera (`facingMode: environment`). An unobtrusive lens switch icon appears only if multiple rear lenses (e.g. macro vs wide) are detected. |
| **Instructions Bottom Sheet** | Expandable card explaining three-step physical placement: 1. Clean spacer contact rim. 2. Place flat against skin. 3. Tap Capture. |
| **Triage Alert Banners** | • **Urgent (Red):** `Possible serious or neoplastic lesion. Refer to dermatologist promptly.` <br>• **Uncertain (Amber):** `Image uncertain or ambiguous. Reposition spacer or consult doctor.` <br>• **Common Condition (Blue):** `Features consistent with [Condition]. Primary care management.` <br>• **Healthy (Green):** `No active lesion detected.` |
| **Heatmap & Contrast Slider** | Shows the preprocessed skin image with a toggleable Grad-CAM++ morphological activation overlay. |

---

## 4. UI States & Field Error Handling

| Scenario | User-Facing Guidance & Remediation |
|---|---|
| **Camera Permission Blocked** | Screen overlay explaining how to enable camera permissions in mobile browser settings. |
| **Tilted Spacer (Light Leak)** | Amber alert: *"Spacer not flat on skin. Light leakage detected. Press evenly against skin."* |
| **Motion Blur (Hand Shake)** | Red alert: *"Blurry capture. Rest arm on flat surface and hold phone steady."* |
| **Analyzing Pipeline** | Pulsing radial progress indicator with label: *"Normalizing colour and analyzing..."* (Buttons disabled). |
| **Offline / Network Interruption** | Offline cache badge: *"Unable to reach server. Check connection or tap Retry."* |
| **Gallery Upload Fallback** | Subordinate link beneath capture button allowing selection of existing photo from mobile camera roll. |

---

## 5. Visual Styling & Mobile Ergonomics

- **Palette:** Clinical Trust theme. Deep navy header (`#0F172A`), clean off-white background (`#F8FAFC`), crisp high-contrast cards (`#FFFFFF`).
- **Severity Semantic Colors:**
  - Urgent Referral: Crimson (`#DC2626`)
  - Review / Common: Cobalt Blue (`#2563EB`)
  - Healthy Skin: Emerald Green (`#059669`)
  - Warning / Uncertain: Warm Amber (`#D97706`)
- **Typography:** System sans-serif (`Inter`, `-apple-system`, `Roboto`) for instantaneous rendering on mobile devices.
- **Accessibility:**
  - Minimum touch targets of $48 \times 48\text{ px}$ across all interactive buttons.
  - WCAG AAA contrast ratio ($\ge 7:1$) on triage advisory text.
  - Dynamic type support for larger font sizes.
