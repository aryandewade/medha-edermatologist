# Medha E-Dermatologist (Mobile Spacer Edition)

AI-assisted skin disease screening using standard smartphones fitted with an inexpensive 3D-printed optical spacer.

| File | Purpose |
|------|---------|
| [prd.md](prd.md) | Product requirements: mobile-first user flow, clinical targets, triage priorities |
| [design.md](design.md) | UI/UX design: single-column mobile layout, spacer framing guide, touch controls |
| [architecture.md](architecture.md) | System architecture: mobile PWA, HTTPS ingress (Cloudflare Tunnel), FastAPI backend |
| [methodology.md](methodology.md) | ML methodology: EfficientNet-B0, mobile augmentations, calibration, triage rules |
| [data-plan.md](data-plan.md) | Clinical datasets (SCIN, Fitzpatrick17k, PAD-UFES-20, DDI), 6-class schema, collection protocol |
| [image-processing.md](image-processing.md) | Preprocessing details: quality gate, colour constancy, lesion localization, glare mitigation |
| [hardware-spacer.md](hardware-spacer.md) | Specifications for the 3D-printed optical spacer, phone cradle, hygiene, and reference patch |
| [api-spec.md](api-spec.md) | REST API contract between mobile PWA and backend (`/predict`, `/health`) |
| [testing-evaluation.md](testing-evaluation.md) | Test strategy, subgroup fairness evaluation, and demo-day hardware checklist |
| [setup-and-deployment.md](setup-and-deployment.md) | Local development setup, Cloudflare Tunnel HTTPS guide, mobile phone configuration |
| [ethics-and-safety.md](ethics-and-safety.md) | Mobile privacy, EXIF/GPS stripping, informed consent, triage ethics |
| [roadmap.md](roadmap.md) | Phased execution plan: mobile spacer first, USB dermatoscope multimodal later |

---

## One-Paragraph Summary
A frontline community health worker opens the web app on their smartphone over secure HTTPS, attaches a 3D-printed optical spacer to the phone's primary camera, places it flush against the patient's skin, and taps **Capture & Analyze**. The backend verifies sharpness and illumination, applies automatic colour constancy, localizes the central skin lesion, and evaluates a calibrated EfficientNet-B0 classifier across 6 classes: **Eczema, Psoriasis, Tinea (Ringworm), Acne, Healthy Skin, and Suspicious Lesions**. High-risk suspicious lesions immediately trigger an urgent specialist referral banner, while ambiguous or low-confidence presentations trigger an honest "uncertain, please consult a dermatologist" advisory.

---

## Read Order for New Team Members
`prd.md` ➔ `hardware-spacer.md` ➔ `image-processing.md` ➔ `architecture.md` ➔ `methodology.md` ➔ `roadmap.md`
