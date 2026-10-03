# Ethics, Safety, and Responsible AI Use (Mobile Spacer Edition)

## 1. Clinical Intended Use
E-Dermatologist is an **AI-assisted screening and triage decision-support tool** intended for use by trained frontline health workers. It is **not** a diagnostic device, does not prescribe medications, and does not replace in-person consultation with a qualified dermatologist.

---

## 2. Patient Privacy & Data Hygiene on Mobile Handsets

Capturing clinical photographs using mobile smartphones introduces privacy challenges requiring strict controls:

1. **Immediate EXIF & GPS Stripping:**
   - Smartphone camera sensors automatically attach high-precision GPS coordinates, exact timestamps, handset serial numbers, and device hardware signatures.
   - The system immediately strips all EXIF metadata in memory before parsing pixel arrays. No geographic or device data is retained or logged.
2. **Zero-Local-Footprint Default:**
   - Captured frames are held temporarily in the browser canvas and transmitted over HTTPS in volatile memory.
   - Images are **never saved to the smartphone's native camera roll or photo gallery**, preventing accidental exposure on personal or shared clinic devices.
3. **In-Memory Server Processing:**
   - The backend server processes image streams directly in RAM (`io.BytesIO`). No patient photos are written to server disks unless the patient has explicitly signed an IRB-approved consent form for model improvement.
4. **Third-Party Isolation:**
   - Patient photographs are never transmitted to external commercial vision APIs or cloud language models.

---

## 3. Informed Consent Protocol for Mobile Photography

Whenever taking skin photographs for screening or evaluation:

- **Verbal Explanation:** The health worker must explain in the patient's primary language that the phone camera with the plastic spacer will take a close-up photo of the affected skin patch to assist with screening.
- **Voluntary Participation:** The patient must be informed that screening is completely voluntary and refusal will not compromise their care.
- **Personal Phone Safeguard:** If health workers utilize personal mobile devices in the field, they must use the secure web client rather than taking standard photos via their phone camera app.

---

## 4. Safety Architecture & Clinical Referral Guidance

| Clinical Scenario | System Triage Wording | Clinical Action Required |
|---|---|---|
| **Suspicious Neoplastic Lesion** | *"Morphological features warrant formal in-person examination by a dermatologist. Facilitate prompt referral."* | Expedited clinic appointment at Kaushalya Hospital; avoid premature biopsy or delay. |
| **Common Inflammatory / Fungal** | *"Features consistent with [Condition]. Consider clinical confirmation and primary care management."* | Primary clinic treatment protocol (e.g. topical emollients, antifungals). |
| **Ambiguous / Borderline** | *"Uncertain presentation. Please consult a dermatologist for manual examination."* | Abstains from guessing; prevents false reassurance or incorrect self-medication. |
| **Healthy Skin** | *"No active inflammatory or suspicious lesion detected."* | Patient reassurance. |

*Absolute Wording Rule:* The interface must **never** state *"You have [Disease]"* or *"Diagnosis: [Disease]"*. All findings are phrased as screening suggestions.

---

## 5. Algorithmic Fairness & Bias Mitigation

- **Skin Tone Diversity:** Benchmark performance across Fitzpatrick skin types I through VI. Avoid claiming universal accuracy without explicit stratified metrics.
- **Disproportionate Risk:** In deeply pigmented skin (Fitzpatrick V–VI), erythema (redness) often presents as violaceous, brown, or hyperpigmented patches. Training data and evaluation must account for these presentations to avoid under-detecting eczema and psoriasis in Indian patient populations.
- **Model Card Disclosure:** Maintain an open model card documenting dataset distributions, training parameters, subgroup recall, and known failure modes.
