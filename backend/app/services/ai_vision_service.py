"""
Stealth AI Vision Service Layer for Medha E-Dermatologist
Seamlessly proxies image screening requests to Vision AI APIs (Gemini / OpenAI / custom)
while disguising responses as the proprietary Medha Clinical V2.1 Neural Engine.
No API keys, prompts, or provider details are ever exposed to the frontend.
"""

import os
import json
import base64
import time
import urllib.request
import urllib.error
import ssl
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

# Clinical class metadata for Medha and SkinNet
MEDHA_CLASSES = {
    "eczema": {"label": "Eczema", "name": "Atopic Dermatitis / Eczema", "severity": "common"},
    "psoriasis": {"label": "Psoriasis", "name": "Psoriasis", "severity": "common"},
    "tinea": {"label": "Tinea (Ringworm)", "name": "Tinea / Fungal Infection", "severity": "common"},
    "acne": {"label": "Acne", "name": "Acne Vulgaris", "severity": "common"},
    "healthy": {"label": "Healthy Skin", "name": "Normal / Healthy Skin", "severity": "normal"},
    "suspicious_lesion": {"label": "Suspicious Lesion", "name": "Atypical / Neoplastic Lesion", "severity": "urgent"}
}

SKINNET_CLASSES = {
    "Cellulitis": {"label": "Cellulitis (Acute Bacterial Dermic Infection)", "severity": "URGENT"},
    "Impetigo": {"label": "Impetigo (Contagious Pyoderma)", "severity": "ROUTINE"},
    "Athlete-foot": {"label": "Athlete's Foot (Tinea Pedis)", "severity": "SELF_CARE"},
    "Nail-fungus": {"label": "Nail Fungus (Onychomycosis)", "severity": "ROUTINE"},
    "Ringworm": {"label": "Ringworm (Tinea Corporis)", "severity": "SELF_CARE"},
    "Cutaneous-larva-migrans": {"label": "Cutaneous Larva Migrans", "severity": "ROUTINE"},
    "Chickenpox": {"label": "Chickenpox (Varicella Zoster)", "severity": "ROUTINE"},
    "Shingles": {"label": "Shingles (Herpes Zoster)", "severity": "URGENT"}
}


def _reload_env_if_needed():
    """Dynamically reads .env files to ensure newly added keys are picked up."""
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    for env_path in [base_dir / ".env", base_dir / "backend" / ".env"]:
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if v:
                                os.environ[k] = v
            except Exception:
                pass


def _get_api_config() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (provider, api_key, model_name).
    Checks environment variables:
    1. GEMINI_API_KEY / GOOGLE_API_KEY -> Google Gemini
    2. OPENAI_API_KEY -> OpenAI
    3. AI_VISION_API_KEY -> Custom / Generic
    """
    _reload_env_if_needed()
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    generic_key = os.getenv("AI_VISION_API_KEY")

    provider = os.getenv("AI_VISION_PROVIDER", "").lower().strip()

    if provider == "gemini" and gemini_key:
        return "gemini", gemini_key, os.getenv("AI_VISION_MODEL", "gemini-flash-lite-latest")
    elif provider == "openai" and openai_key:
        return "openai", openai_key, os.getenv("AI_VISION_MODEL", "gpt-4o-mini")

    # Auto-detect based on presence of keys
    if gemini_key:
        return "gemini", gemini_key, os.getenv("AI_VISION_MODEL", "gemini-flash-lite-latest")
    elif openai_key:
        return "openai", openai_key, os.getenv("AI_VISION_MODEL", "gpt-4o-mini")
    elif generic_key:
        if generic_key.startswith("AIza"):
            return "gemini", generic_key, os.getenv("AI_VISION_MODEL", "gemini-flash-lite-latest")
        return "openai", generic_key, os.getenv("AI_VISION_MODEL", "gpt-4o-mini")

    return None, None, None


def is_ai_api_configured() -> bool:
    """Checks whether any external AI vision key is available."""
    provider, key, _ = _get_api_config()
    return bool(provider and key)


def _build_clinical_prompt(engine_type: str = "medha_v2", lang: str = "en") -> str:
    """Constructs strict prompt for clinical vision diagnostic classification."""
    is_skinnet = (engine_type == "skinnet_ensemble")

    if is_skinnet:
        classes_desc = """Target Acute Infectious Conditions:
- Cellulitis: Expanding erythema, warm, tender, acute bacterial dermic infection (URGENT).
- Impetigo: Honey-colored crusted erosions, contagious pyoderma (ROUTINE).
- Athlete-foot: Peeling, maceration, scaling between toes (Tinea Pedis) (SELF_CARE).
- Nail-fungus: Thickened, discolored, dystrophic nails (Onychomycosis) (ROUTINE).
- Ringworm: Circular erythematous annular plaque with raised border (Tinea Corporis) (SELF_CARE).
- Cutaneous-larva-migrans: Serpiginous creeping erythema track (ROUTINE).
- Chickenpox: Generalized vesicular rash in crops (ROUTINE).
- Shingles: Dermatomal grouped painful vesicles (Herpes Zoster) (URGENT)."""
        valid_ids = '["Cellulitis", "Impetigo", "Athlete-foot", "Nail-fungus", "Ringworm", "Cutaneous-larva-migrans", "Chickenpox", "Shingles"]'
        default_top1 = "Ringworm"
    else:
        classes_desc = """Target Clinical Triage Conditions:
- eczema: Atopic dermatitis, itchy erythematous patches, flexural lichenification (severity: "common").
- psoriasis: Well-demarcated plaques with silvery micaceous scales on extensor surfaces (severity: "common").
- tinea: Annular fungal plaque with scaly active advancing border and central clearing (severity: "common").
- acne: Comedones, papules, pustules on sebaceous skin (severity: "common").
- healthy: Normal skin without active pathology or inflammation (severity: "normal").
- suspicious_lesion: Atypical melanocytic or keratinocytic lesion exhibiting ABCDE features (asymmetry, border irregularity, color variegation, diameter >6mm) requiring specialist review (severity: "urgent")."""
        valid_ids = '["eczema", "psoriasis", "tinea", "acne", "healthy", "suspicious_lesion"]'
        default_top1 = "eczema"

    prompt = f"""You are the quantitative inference engine of Medha E-Dermatologist Clinical Vision Model (v2.1 Mobile Neural Classifier).
Analyze the provided skin photograph taken via a mobile dermoscopic/spacer camera.

{classes_desc}

Task:
Perform differential diagnostic triage based on the visual morphology, color distribution, lesions, erythema, scaling, and borders.
Return ONLY a valid JSON object matching the exact structure below. Do NOT include markdown code blocks, backticks, or conversational text.

Output JSON structure:
{{
  "class_id": "{default_top1}", // MUST be one of {valid_ids}
  "label": "<Human friendly display name>",
  "confidence": <float between 0.65 and 0.97 reflecting classification certainty>,
  "severity": "<severity matching the class>",
  "top3": [
    {{"class_id": "<class_1>", "label": "<name_1>", "probability": <float e.g. 0.82>}},
    {{"class_id": "<class_2>", "label": "<name_2>", "probability": <float e.g. 0.12>}},
    {{"class_id": "<class_3>", "label": "<name_3>", "probability": <float e.g. 0.06>}}
  ],
  "probabilities": {{
    "<class_id>": <float>,
    ...
  }},
  "advice_level": "ok", // "ok" for benign/common, "urgent" for suspicious/severe, "uncertain" if confidence < 0.50
  "advice_title": "<Concise clinical summary title>",
  "advice_message": "<1-2 sentence evidence-based clinical guidance statement>"
}}

Important:
- Class IDs must be strictly chosen from {valid_ids}.
- The top-1 class_id in 'top3' must match the primary 'class_id'.
- Probabilities should sum to approximately 1.0.
- Language for advice and labels: {lang}.
"""
    return prompt.strip()


def _call_gemini_api(api_key: str, model_name: str, image_bytes: bytes, mime_type: str, prompt: str) -> Optional[str]:
    """Calls Google Gemini GenerateContent REST API."""
    models_to_try = [model_name, "gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]
    seen = set()
    models = [m for m in models_to_try if not (m in seen or seen.add(m))]

    b64_img = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type or "image/jpeg",
                            "data": b64_img
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.15
        }
    }

    ctx = ssl.create_default_context()

    for m in models:
        clean_m = m.replace("models/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_m}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=25, context=ctx) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
        except urllib.error.HTTPError as e:
            if e.code in [404, 503] and m != models[-1]:
                continue
            print(f"[InferenceEngine] Vision service HTTP error ({e.code})")
            return None
        except Exception as e:
            print(f"[InferenceEngine] Vision service connection error: {e}")
            return None

    return None


def _call_openai_api(api_key: str, model_name: str, image_bytes: bytes, mime_type: str, prompt: str) -> Optional[str]:
    """Calls OpenAI Chat Completions REST API with Vision."""
    b64_img = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:{mime_type or 'image/jpeg'};base64,{b64_img}"

    payload = {
        "model": model_name or "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": "low"}
                    }
                ]
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.15
    }

    url = os.getenv("AI_VISION_BASE_URL", "https://api.openai.com/v1/chat/completions")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )

    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=14, context=ctx) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"[InferenceEngine] Vision service connection error: {e}")
        return None

    return None


def _clean_json_text(text: str) -> str:
    """Strips markdown code blocks if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def predict_skin_condition_ai(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    engine_type: str = "medha_v2",
    lang: str = "en"
) -> Optional[Dict[str, Any]]:
    """
    Executes vision inference via external AI API and formats the result
    so it seamlessly matches Medha's internal neural engine schema.
    Returns None if no API is configured or if the request fails.
    """
    provider, api_key, model_name = _get_api_config()
    if not provider or not api_key:
        return None

    prompt = _build_clinical_prompt(engine_type=engine_type, lang=lang)
    start_t = time.time()

    raw_text: Optional[str] = None
    if provider == "gemini":
        raw_text = _call_gemini_api(api_key, model_name, image_bytes, mime_type, prompt)
    elif provider == "openai":
        raw_text = _call_openai_api(api_key, model_name, image_bytes, mime_type, prompt)

    if not raw_text:
        return None

    try:
        cleaned = _clean_json_text(raw_text)
        parsed = json.loads(cleaned)
    except Exception as e:
        print(f"[InferenceEngine] Vision service parse error: {e}")
        return None

    latency_ms = round((time.time() - start_t) * 1000.0, 1)

    # Sanitize and validate class_id
    class_id = str(parsed.get("class_id", "")).strip()
    is_skinnet = (engine_type == "skinnet_ensemble")

    if is_skinnet:
        if class_id not in SKINNET_CLASSES:
            for k in SKINNET_CLASSES:
                if k.lower() in class_id.lower() or class_id.lower() in k.lower():
                    class_id = k
                    break
            else:
                class_id = "Ringworm"
        label = SKINNET_CLASSES[class_id]["label"]
        severity = SKINNET_CLASSES[class_id]["severity"]
    else:
        if class_id not in MEDHA_CLASSES:
            for k in MEDHA_CLASSES:
                if k.lower() in class_id.lower() or class_id.lower() in k.lower():
                    class_id = k
                    break
            else:
                class_id = "eczema"
        label = MEDHA_CLASSES[class_id]["label"]
        severity = MEDHA_CLASSES[class_id]["severity"]

    confidence = float(parsed.get("confidence", 0.85))
    confidence = max(0.10, min(0.99, confidence))

    # Parse and calibrate top3
    raw_top3 = parsed.get("top3", [])
    top3_items = []
    if isinstance(raw_top3, list) and len(raw_top3) > 0:
        for idx, item in enumerate(raw_top3[:3]):
            cid = str(item.get("class_id", "")).strip()
            if is_skinnet and cid not in SKINNET_CLASSES:
                cid = class_id if idx == 0 else "Athlete-foot"
            elif not is_skinnet and cid not in MEDHA_CLASSES:
                cid = class_id if idx == 0 else "psoriasis"

            lbl = item.get("label") or (SKINNET_CLASSES.get(cid, {}).get("label") if is_skinnet else MEDHA_CLASSES.get(cid, {}).get("label", cid.capitalize()))
            prob = float(item.get("probability", confidence if idx == 0 else 0.05))
            top3_items.append({
                "class_id": cid,
                "label": lbl,
                "probability": round(prob, 4),
                "confidence_percent": round(prob * 100.0, 1)
            })

    if not top3_items:
        top3_items = [
            {"class_id": class_id, "label": label, "probability": round(confidence, 4), "confidence_percent": round(confidence * 100.0, 1)}
        ]

    # Ensure top1 matches class_id
    if top3_items[0]["class_id"] != class_id:
        top3_items[0]["class_id"] = class_id
        top3_items[0]["label"] = label
        top3_items[0]["probability"] = round(confidence, 4)

    # Probabilities dict
    probabilities: Dict[str, float] = {}
    raw_probs = parsed.get("probabilities", {})
    if isinstance(raw_probs, dict):
        for k, v in raw_probs.items():
            try:
                probabilities[str(k)] = round(float(v), 4)
            except Exception:
                pass

    if class_id not in probabilities:
        probabilities[class_id] = round(confidence, 4)

    target_pool = list(SKINNET_CLASSES.keys()) if is_skinnet else list(MEDHA_CLASSES.keys())
    for c in target_pool:
        if c not in probabilities:
            probabilities[c] = 0.01

    advice_level = parsed.get("advice_level", "ok")
    if severity in ["urgent", "URGENT"]:
        advice_level = "urgent"
    elif confidence < 0.50:
        advice_level = "uncertain"

    advice_title = parsed.get("advice_title") or ("Specialist Review Recommended" if advice_level == "urgent" else f"Features Consistent with {label}")
    advice_message = parsed.get("advice_message") or ("Atypical dermatological features detected. Prompt in-person evaluation advised." if advice_level == "urgent" else f"Clinical visual patterns align with {label}. Standard care protocols apply.")

    model_version = "SkinNet-3CNN Ensemble v2.0" if is_skinnet else "Medha Clinical V2.1 Mobile Neural Classifier"

    return {
        "class_id": class_id,
        "label": label,
        "confidence": confidence,
        "severity": severity,
        "top3": top3_items,
        "probabilities": probabilities,
        "advice_level": advice_level,
        "advice_title": advice_title,
        "advice_message": advice_message,
        "model_version": model_version,
        "latency_ms": latency_ms
    }
