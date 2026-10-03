"""
Interactive Symptom Verification & Refinement Engine
Adapted from SkinNet-Analyzer with extensions for Medha's clinical dermatosis ontology.

Uses Bayesian likelihood updating to refine visual AI predictions with patient-reported symptoms,
prevents visual false positives, and computes clinical severity (Mild / Moderate / Severe).
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Union

# Canonical symptom equivalences
EQUIVALENT_SYMPTOMS = {
    # Systemic Symptoms
    "fever": ["fever", "chills", "high temperature"],
    "fatigue": ["tiredness", "fatigue", "malaise"],
    "pain": ["pain", "burning pain", "nerve pain", "painful swelling", "tenderness"],
    
    # Sensations
    "burning/itching": ["itching", "burning", "pruritus", "stinging"],
    
    # Lesion Morphology
    "sores": ["sores", "open sores", "ulcerations"],
    "crusting": ["crusting", "honey-colored crusts", "oozing crusts"],
    "swelling": ["swelling", "painful swelling", "inflammation", "edema"],
    "blisters": ["blisters", "fluid-filled blisters", "vesicles"],
    "warm skin": ["warm skin", "heat radiating from skin"],
    
    # Visual Patterns
    "redness": ["redness", "erythema"],
    "ring_pattern": ["red ring-shaped patch", "annular rash", "raised circular border", "thread/ring like pattern"],
    "creeping_track": ["red lines on skin", "serpiginous track", "creeping track"],
    "skin_texture_changes": ["peeling skin", "scaly skin", "cracks", "dry patches"],
    "silvery_scales": ["silvery/white scales", "micaceous plaque scales"],
    "nail_changes": ["thickened nails", "nail discoloration", "brittle nails", "onycholysis"],
    "bad_odor": ["bad odor", "foul smell from area"],
    "blackheads_pustules": ["comedones", "blackheads", "whiteheads", "pustules", "pimples"],
    "abcde_features": ["asymmetry", "irregular borders", "color variegation", "diameter >6mm", "evolving mole"]
}

# Disease symptom knowledge base (SkinNet 8 classes + Medha classes)
SYMPTOM_MAPPING: Dict[str, List[str]] = {
    # --- SkinNet-8 Classes ---
    "Cellulitis": ["redness", "swelling", "warm skin", "pain", "fever"],
    "Impetigo": ["sores", "burning/itching", "blisters", "crusting"],
    "Ringworm": ["ring_pattern", "burning/itching", "skin_texture_changes", "swelling"],
    "Cutaneous-larva-migrans": ["burning/itching", "creeping_track", "pain", "swelling"],
    "Chickenpox": ["fever", "fatigue", "burning/itching", "blisters"],
    "Shingles": ["pain", "burning/itching", "blisters"],
    "Athlete-foot": ["burning/itching", "skin_texture_changes", "blisters"],
    "Nail-fungus": ["nail_changes", "bad_odor", "pain"],

    # --- Medha Primary Classes ---
    "Eczema": ["burning/itching", "skin_texture_changes", "redness", "crusting"],
    "Psoriasis": ["silvery_scales", "skin_texture_changes", "redness", "burning/itching"],
    "Tinea": ["ring_pattern", "burning/itching", "skin_texture_changes", "redness"],
    "Acne": ["blackheads_pustules", "redness", "pain"],
    "Suspicious Lesion": ["abcde_features", "pain"]
}

# User-friendly question texts
SYMPTOM_QUESTIONS: Dict[str, str] = {
    "fever": "Have you had a fever, chills, or felt generally unwell recently?",
    "fatigue": "Are you experiencing noticeable fatigue or muscle weakness?",
    "pain": "Is the affected skin tender, throbbing, or causing significant pain?",
    "burning/itching": "Does the affected area itch intensely or feel like it is burning?",
    "sores": "Are there open sores, weeping lesions, or skin abrasions present?",
    "crusting": "Is there crusting, scabbing, or golden/honey-colored fluid drying on the skin?",
    "swelling": "Is the area visibly swollen or inflamed compared to surrounding skin?",
    "blisters": "Do you see small or clustered fluid-filled blisters (vesicles)?",
    "warm skin": "Does the skin feel noticeably hot or warm to the touch?",
    "redness": "Is there noticeable redness spreading over the area?",
    "ring_pattern": "Does the rash have an expanding circular ring shape with clearer skin inside?",
    "creeping_track": "Are there wavy, winding red lines or a creeping thread-like track on the skin?",
    "skin_texture_changes": "Is the skin cracked, flaking, peeling, or unusually dry and thickened?",
    "silvery_scales": "Are there silvery-white, flaky scales over well-defined raised plaques?",
    "nail_changes": "Are your nails discolored (yellow/brown), thickened, or crumbly?",
    "bad_odor": "Is there a noticeable unpleasant odor coming from the affected area?",
    "blackheads_pustules": "Are there blackheads, whiteheads, or pus-filled pimples in the area?",
    "abcde_features": "Has this spot/mole recently changed in size, shape, color, or does it have ragged edges?"
}

# Assumed probability that patient's answer is accurate
ANSWER_RELIABILITY = 0.70
OUT_OF_CLASS_PHOTO_CONFIDENCE = 0.40


def get_symptom_questions(candidate_diseases: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Returns relevant symptom questions for the top candidate diseases.
    """
    unique_symptoms = set()
    valid_diseases = []

    for d in candidate_diseases:
        # Normalize name matching
        key = None
        for k in SYMPTOM_MAPPING:
            if k.lower() in d.lower() or d.lower() in k.lower():
                key = k
                break
        if key:
            valid_diseases.append(key)
            unique_symptoms.update(SYMPTOM_MAPPING[key])

    # If no mapping found, return common dermatological questions
    if not unique_symptoms:
        unique_symptoms = {"burning/itching", "pain", "redness", "skin_texture_changes"}

    questions = {
        symptom: SYMPTOM_QUESTIONS.get(symptom, f"Do you notice {symptom.replace('_', ' ')}?")
        for symptom in unique_symptoms
    }

    return questions, valid_diseases


def process_symptom_responses(
    disease_keys: List[str],
    answers: Dict[str, Union[str, int, bool]],
    probabilities: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Refines visual predictions using patient symptom responses via Bayesian log-likelihood scoring.
    Computes disease confirmation, severity score, and matching statistics.
    """
    if not disease_keys:
        return {"error": "No disease keys provided"}

    if not probabilities or len(probabilities) != len(disease_keys):
        probabilities = [1.0 / len(disease_keys)] * len(disease_keys)

    # Convert answers to boolean (1/true = Yes, 0/false = No)
    answered: Dict[str, bool] = {}
    for symptom, val in answers.items():
        if isinstance(val, bool):
            answered[symptom] = val
        elif str(val).lower() in ("1", "true", "yes"):
            answered[symptom] = True
        elif str(val).lower() in ("0", "false", "no"):
            answered[symptom] = False

    log_right = math.log(ANSWER_RELIABILITY)
    log_wrong = math.log(1.0 - ANSWER_RELIABILITY)

    scores: Dict[str, float] = {}
    normalized_keys = []

    for disease, prob in zip(disease_keys, probabilities):
        # Find matching key in mapping
        match_key = disease
        for k in SYMPTOM_MAPPING:
            if k.lower() in disease.lower() or disease.lower() in k.lower():
                match_key = k
                break

        normalized_keys.append(match_key)
        disease_symptoms = SYMPTOM_MAPPING.get(match_key, [])

        agree = sum((symptom in disease_symptoms) == user_yes for symptom, user_yes in answered.items())
        total_answered = len(answered)
        disagree = total_answered - agree

        # Bayesian update: log(P(photo)) + agree * log(p) + disagree * log(1-p)
        scores[disease] = math.log(prob + 1e-9) + agree * log_right + disagree * log_wrong

    # Rank diseases based on updated posterior scores
    confirmed_disease = max(scores, key=scores.get)
    confirmed_idx = disease_keys.index(confirmed_disease)
    photo_confidence = probabilities[confirmed_idx]

    # Map to symptom list
    norm_confirmed = normalized_keys[confirmed_idx]
    confirmed_symptoms = SYMPTOM_MAPPING.get(norm_confirmed, [])

    # Matched symptoms count
    matched = sum(1 for sym in confirmed_symptoms if answered.get(sym, False) is True)
    total_expected = len(confirmed_symptoms)
    severity_ratio = (matched / total_expected) if total_expected > 0 else 0.0

    # Determine clinical severity
    if severity_ratio < 0.25:
        severity = "Out of Class" if photo_confidence < OUT_OF_CLASS_PHOTO_CONFIDENCE else "Mild"
    elif severity_ratio <= 0.50:
        severity = "Mild"
    elif severity_ratio < 0.75:
        severity = "Moderate"
    else:
        severity = "Severe"

    # Softmax on scores for posterior distribution percentage
    max_s = max(scores.values())
    exp_scores = {k: math.exp(v - max_s) for k, v in scores.items()}
    sum_exp = sum(exp_scores.values())
    posterior_probs = {k: round(v / sum_exp, 4) for k, v in exp_scores.items()}

    confirmed_prob = posterior_probs.get(confirmed_disease, photo_confidence)

    return {
        "confirmed_disease": confirmed_disease,
        "severity": severity,
        "severity_percentage": round(severity_ratio * 100, 1),
        "symptoms_matched": f"{matched}/{total_expected}",
        "photo_confidence": round(photo_confidence, 4),
        "confirmed_confidence": round(confirmed_prob, 4),
        "posterior_probabilities": posterior_probs,
        "disease_scores": {k: round(v, 3) for k, v in scores.items()},
        "is_reordered": confirmed_disease != disease_keys[0],
        "original_top1": disease_keys[0]
    }
