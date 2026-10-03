# Hardware Specification: 3D-Printed Optical Spacer

## 1. Purpose & Physical Rationale

Standard smartphone cameras cannot reliably photograph close-up skin lesions because:
1. **Uncontrolled Focal Distance:** Hand tremor and micro-movements cause autofocus hunting, yielding blurred captures.
2. **Variable Illumination & Glare:** Direct camera flash causes harsh specular reflection (white hot spots) that obscures lesion pigment networks and texture.
3. **Scale Ambiguity:** Without a physical scale reference, machine learning models cannot discern the true physical dimensions of a lesion (e.g., a 2 mm macule vs. a 15 mm patch).

The **3D-Printed Optical Spacer** physically clips onto the smartphone to establish a **fixed, repeatable optical focal plane**, control ambient illumination, and present a calibrated scale and colour reference in every capture.

---

## 2. Spacer Design Specifications

```
             [Smartphone Body]
        ═══════════════════════════
                 [Lens]
                   │  (Aperture Ø 14 mm)
           ┌───────┴───────┐
           │   Top Collar  │
           │               │
           │  Conical Tube │  ◄── Matte Black Interior
           │  (Height:     │      (Roughness Ra > 3.2 µm)
           │   35 mm)      │
       ┌───┴───────────────┴───┐
       │   Contact Ring Flange │  ◄── Radius 24 mm
       │ [Ref Patch] [5mm Bar] │
       └───────────────────────┘
        ═══════════════════════════
               [Patient Skin]
```

### Physical Dimensions
- **Focal Height ($H$):** $35.0\text{ mm}$ (optimal for standard smartphone wide lenses having a minimum focus distance of $50\text{–}70\text{ mm}$ with $1.5\times$ digital macro framing, or $35\text{ mm}$ for lenses with near-macro autofocus).
- **Optional Stackable Rings:** Modular $5\text{ mm}$ snap-fit extension rings to adjust total focal height from $30\text{ mm}$ to $50\text{ mm}$ depending on the specific handset camera module.
- **Top Camera Aperture:** $14.0\text{ mm}$ diameter, oversized to clear multi-lens camera bumps without vignetting.
- **Base Skin Contact Diameter:** $48.0\text{ mm}$ outer diameter ($38.0\text{ mm}$ inner clear aperture) to capture adequate lesion margins.

---

## 3. Optical & Lighting Control

- **Matte-Black Interior Wall:**
  - The interior conical surface must feature a **dead-matte black finish** to eliminate internal specular reflections and barrel flaring.
  - Sliced with internal concentric micro-grooves ($0.2\text{ mm}$ step height) to trap grazing-angle light rays.
- **Ambient Lighting Ports:**
  - Includes two angled, translucent diffuser side windows ($45^\circ$ angle of incidence) covered with white diffusion film.
  - Allows ambient clinic room light to illuminate the lesion softly and uniformly while completely blocking direct glare and shadowing from the phone handset.
- **Flash Lockout:** The spacer collar physically occludes the smartphone's built-in LED flash to prevent accidental flash blinding.

---

## 4. Materials, Hygiene & Biocompatibility

- **Filament Material:** **PETG** (Polyethylene Terephthalate Glycol) or **Medical-Grade PLA**.
  - High chemical resistance against alcohol and disinfectants.
  - Biocompatible for transient non-broken skin contact (ISO 10993 compliant).
- **Sanitization Protocol:**
  - The skin-contact flange must be thoroughly wiped with a **70% Isopropyl Alcohol (IPA)** disinfectant wipe before and after every patient contact.
  - PETG maintains dimensional integrity and does not craze or degrade after repeated alcohol wipe exposure.
- **Wall Thickness:** $2.0\text{ mm}$ minimum to guarantee mechanical rigidity and prevent ambient light bleeding through the plastic walls.

---

## 5. Calibrated Reference Patch Integration

Mounted flush along the interior lower rim of the contact aperture:
1. **18% Neutral Gray Card Segment:** Provides ground-truth neutral reflectance for automatic white balance and colour constancy algorithms.
2. **90% Reflective White Patch:** Defines maximum dynamic range and detects LED overexposure.
3. **5 mm High-Contrast Scale Bar:** Alternating black/white $1.0\text{ mm}$ millimeter ticks enabling pixel-to-millimeter spatial calibration.

---

## 6. Phone Cradle & Handset Compatibility

The spacer mounts to the handset via an adjustable spring-loaded clamping bracket or model-specific snap-on phone case:

| Phone Model | Primary Lens Position | Min. Focus Distance | Recommended Spacer Height |
|---|---|---|---|
| **Samsung Galaxy S22 / S23 / S24** | Center-left top lens (50MP 1x wide) | $45\text{ mm}$ | $35\text{ mm}$ (with $1.2\times$ crop) |
| **iPhone 13 / 14 / 15 (Standard)** | Bottom-left lens (1x wide) | $50\text{ mm}$ | $35\text{ mm}$ |
| **iPhone 13 / 14 / 15 Pro** | Bottom-left lens (locked to 1x wide) | $40\text{ mm}$ (disable macro auto-switch) | $35\text{ mm}$ |
| **OnePlus 11 / 12 / Nord** | Upper central circular cluster | $50\text{ mm}$ | $40\text{ mm}$ |
| **Google Pixel 7 / 8** | Horizontal camera bar (left lens) | $45\text{ mm}$ | $35\text{ mm}$ |

---

## 7. 3D Print Settings (FDM / SLA)

```ini
Layer Height: 0.16 mm (for smooth internal micro-grooves)
Infill Density: 100% (Solid, to block light transmission)
Perimeters / Shells: 4
Material: Black PETG (Matte preferred)
Nozzle Temp: 235°C
Bed Temp: 75°C
Supports: None required (designed with 45° overhang angle)
Post-Processing: Wipe interior with isopropyl alcohol; apply matte black acrylic spray if glossy filament was used.
```
