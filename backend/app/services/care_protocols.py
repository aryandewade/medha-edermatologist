"""
Clinical Care Protocols and Action Guidance
Ported and adapted from SkinNet-Analyzer for clinical triage in primary dermatology.
Provides:
- External symptoms (visual presentation)
- Internal / systemic symptoms
- Evidence-based self-care and medical steps
- Urgent red flag alerts ("Consult a doctor urgently if...")
"""

from typing import Dict, Any

CARE_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    # --- SkinNet-8 Infectious Conditions ---
    "Cellulitis": {
        "external": [
            "Rapidly expanding, red, swollen, warm, and tender area of skin",
            "Tight, glossy skin texture; occasionally blistering or skin dimpling (peau d'orange)"
        ],
        "internal": [
            "Fever, chills, and rigors",
            "General malaise, fatigue, and headache",
            "Swollen, tender regional lymph nodes (e.g., groin, armpit)"
        ],
        "care": [
            "Seek urgent medical evaluation - cellulitis requires systemic prescription antibiotics",
            "Complete the entire antibiotic regimen strictly as prescribed, even after visible improvement",
            "Elevate the affected limb above heart level to reduce swelling and edema",
            "Mark the border of erythema with a surgical marker/pen to monitor whether it is expanding or regressing",
            "Keep the affected skin clean, cool, and lightly covered with sterile non-adherent dressing"
        ],
        "urgent": "redness spreads rapidly, high fever develops (>38.5°C), red streaks travel toward the heart, or severe throbbing pain occurs."
    },

    "Impetigo": {
        "external": [
            "Erythematous sores that rupture to produce classic honey-colored, golden crusts",
            "Small fluid-filled blisters (bullae) commonly around the nostrils, mouth, or limbs"
        ],
        "internal": [
            "Usually no systemic signs",
            "Occasionally mild localized lymphadenopathy or low-grade fever"
        ],
        "care": [
            "Consult a healthcare clinic for prescription topical (mupirocin) or oral antibiotics",
            "Gently soak and cleanse crusts with warm water and antiseptic soap; pat dry with clean paper towels",
            "Keep fingernails trimmed short to avoid scratching and autoinoculation",
            "Maintain strict hygiene: do not share towels, pillows, or washcloths; impetigo is highly contagious"
        ],
        "urgent": "lesions spread widely across the body, high fever occurs, or sores do not improve after 3 days of therapy."
    },

    "Ringworm": {
        "external": [
            "Annular (circular) erythematous plaque with a raised, scaly advancing border",
            "Central clearing (skin looks normal or pale inside the ring)",
            "Intense localized itching and flaking"
        ],
        "internal": ["Generally none"],
        "care": [
            "Apply over-the-counter topical antifungal cream (terbinafine, clotrimazole, or ketoconazole) twice daily",
            "Extend cream application 2 cm beyond the visible rash edge for at least 2 weeks after visual clearance",
            "Keep the affected area clean, dry, and aerated",
            "Wash clothing and bedding in hot water; inspect household pets as they are common transmission reservoirs"
        ],
        "urgent": "the infection involves the scalp (tinea capitis requires oral antifungals), nails, becomes superinfected with bacteria (pus), or fails to improve in 14 days."
    },

    "Athlete-foot": {
        "external": [
            "Macerated, soggy white skin or erythematous cracks between the toes (interdigital)",
            "Scaly, dry moccasin-pattern flaking on the soles and sides of the feet",
            "Burning, stinging, and itchiness; occasional friction blisters"
        ],
        "internal": ["None"],
        "care": [
            "Apply topical antifungal spray or cream thoroughly between toes and across soles twice daily",
            "Wash feet daily and dry meticulously between each toe with an individual towel",
            "Wear moisture-wicking cotton or wool socks and change them promptly if damp",
            "Disinfect footwear and avoid walking barefoot in communal showers, locker rooms, or pool decks"
        ],
        "urgent": "skin becomes intensely painful, warm, oozing pus, red streaks spread up the ankle, or you have diabetes/peripheral vascular disease."
    },

    "Nail-fungus": {
        "external": [
            "Thickened, brittle, crumbly, or distorted nail plates (onychodystrophy)",
            "Subungual hyperkeratosis with yellow, brown, or white discoloration",
            "Foul or sour odor under the affected nail bed"
        ],
        "internal": ["None; may cause localized discomfort when wearing shoes"],
        "care": [
            "Consult a dermatologist or podiatrist; oral antifungals (terbinafine) or medicated lacquers are needed",
            "Keep nails trimmed short and filed down straight across",
            "Avoid sharing nail clippers, files, or emery boards with family members",
            "Treat concurrent athlete's foot simultaneously to prevent re-infection"
        ],
        "urgent": "the nail bed becomes inflamed, oozes pus, is accompanied by spreading periungual erythema, or you have diabetes."
    },

    "Cutaneous-larva-migrans": {
        "external": [
            "Characteristic winding, serpiginous (snake-like) raised erythematous track that advances 1-2 cm daily",
            "Intense, disruptive pruritus; occasional localized vesiculation"
        ],
        "internal": ["Rarely systemic; may experience sleep loss due to nocturnal itching"],
        "care": [
            "Consult a physician or travel medicine specialist for prescription oral anthelmintics (albendazole/ivermectin)",
            "Apply topical antipruritic lotion (calamine) or cool compresses to calm itching",
            "Refrain from scratching to prevent secondary bacterial superinfection",
            "Avoid walking barefoot or lying directly on damp soil or sand where dogs/cats roam"
        ],
        "urgent": "secondary bacterial infection develops (pus, spreading redness, swelling, warmth, or fever)."
    },

    "Chickenpox": {
        "external": [
            "Widespread rash developing in crops: macular spots progressing to papules, fluid-filled 'dewdrop' vesicles, then scabs",
            "Intense generalized pruritus; lesions present in multiple stages simultaneously"
        ],
        "internal": [
            "Prodromal fever, headache, sore throat, and loss of appetite",
            "Generalized malaise and body aches"
        ],
        "care": [
            "Isolate at home until ALL vesicles have completely dried and formed firm scabs (usually 7-10 days)",
            "Use calamine lotion, oatmeal baths, and cool damp compresses to soothe severe itching",
            "Manage fever with paracetamol; NEVER administer aspirin to children or adolescents (risk of Reye's syndrome)",
            "Keep patient hydrated and fingernails trimmed smooth to prevent excoriation scars"
        ],
        "urgent": "the patient is an infant, pregnant, immunocompromised, develops shortness of breath, confusion, extreme lethargy, or blisters become inflamed with surrounding red swelling."
    },

    "Shingles": {
        "external": [
            "Painful, clustered vesicular eruption distributed strictly in a unilateral dermatomal band (one side of body or face)",
            "Preceded by tingling, burning, or allodynia (pain to light touch) 1-5 days before lesions emerge"
        ],
        "internal": [
            "Sharp, burning, or electric-shock-like neuropathic pain (postherpetic neuralgia risk)",
            "Mild fever, headache, and photophobia"
        ],
        "care": [
            "Consult a physician immediately: oral antiviral therapy (acyclovir, valacyclovir) is most effective when initiated within 72 hours of rash onset",
            "Keep the rash clean, dry, and loosely covered with non-stick sterile gauze",
            "Avoid direct skin contact with individuals who have never had chickenpox, pregnant women, and immunocompromised individuals",
            "Use prescribed neuropathic analgesics or cool compresses for pain relief"
        ],
        "urgent": "rash involves the tip of the nose or forehead near the eye (herpes zoster ophthalmicus - danger of vision loss!), neck stiffness, or motor weakness."
    },

    # --- Medha Primary Clinical Triage Classes ---
    "Eczema": {
        "external": [
            "Dry, lichenified (thickened), scaly erythematous plaques in skin flexures (elbow creases, behind knees, wrists)",
            "Excoriations, weeping micro-vesicles during acute flare-ups"
        ],
        "internal": ["None; sleep disturbance due to intractable nocturnal itching"],
        "care": [
            "Apply thick ceramide-rich emollient moisturizers at least twice daily, especially within 3 minutes after bathing",
            "Use short, lukewarm showers with gentle fragrance-free non-soap syndet cleansers",
            "Apply prescribed mild topical corticosteroid or calcineurin inhibitor during active inflammatory flare-ups",
            "Wear soft, breathable cotton clothing and minimize exposure to wool, harsh detergents, and known allergens"
        ],
        "urgent": "skin becomes intensely painful, oozing golden fluid, or develops clusters of punched-out fever blisters (eczema herpeticum requires emergency antiviral care)."
    },

    "Psoriasis": {
        "external": [
            "Well-demarcated, raised, salmon-pink plaques capped with thick, adherent micaceous silvery-white scales",
            "Predilection for extensor surfaces (elbows, knees, sacrum, scalp)",
            "Positive Auspitz sign (pinpoint bleeding when scales are gently detached)"
        ],
        "internal": [
            "Joint stiffness or morning arthralgia (psoriatic arthritis in up to 30% of patients)"
        ],
        "care": [
            "Apply prescribed topical vitamin D analogues, keratolytics (salicylic acid), and corticosteroids as directed",
            "Maintain consistent daily skin hydration with thick barrier creams to reduce scaling and fissuring",
            "Engage in moderate, controlled natural sunlight exposure while avoiding sunburns (Koebner phenomenon)",
            "Consult a dermatologist for systemic therapy or biologics if disease is extensive or causing joint pain"
        ],
        "urgent": "widespread rapid eruption of pustules (generalized pustular psoriasis) or skin becomes completely red and shedding (erythroderma) accompanied by chills/fever."
    },

    "Tinea": {
        "external": [
            "Expanding circular lesion with an active, raised, vesicular or scaly border and clearer center",
            "Mild to moderate localized pruritus and peeling"
        ],
        "internal": ["None"],
        "care": [
            "Use topical antifungal cream (terbinafine, clotrimazole) applied 2 cm beyond the border for 2-4 weeks",
            "Avoid topical steroids (hydrocortisone, betamethasone) which worsen fungal infections (tinea incognito)",
            "Keep the area clean, cool, and well-ventilated; avoid tight non-breathable synthetic clothing"
        ],
        "urgent": "lesion involves the scalp, beard area, or begins oozing pus indicative of bacterial superinfection."
    },

    "Acne": {
        "external": [
            "Open (blackheads) and closed (whiteheads) comedones",
            "Inflammatory papules, pustules, and deep nodules on face, chest, or upper back"
        ],
        "internal": ["None"],
        "care": [
            "Wash face twice daily with a mild foaming cleanser; avoid harsh abrasive scrubs",
            "Incorporate evidence-based topicals: benzoyl peroxide (2.5-5%), salicylic acid, or adapalene",
            "Do not pick, squeeze, or pop acne lesions to prevent permanent scarring and post-inflammatory hyperpigmentation",
            "Use non-comedogenic, oil-free moisturizers and daily broad-spectrum SPF 30+ sunscreen"
        ],
        "urgent": "large, painful, interconnected fluctuant cysts or nodulocystic acne that causes deep scarring."
    },

    "Suspicious Lesion": {
        "external": [
            "Asymmetric pigmented macule, papule, or plaque with irregular, notched borders",
            "Color variation (shades of brown, black, blue, red, or white within the same lesion)",
            "Diameter >6 mm or any lesion displaying rapid growth, ulceration, or spontaneous bleeding"
        ],
        "internal": ["Generally none in early stages"],
        "care": [
            "Schedule an in-person dermoscopic examination and clinical evaluation with a dermatologist promptly",
            "Do NOT attempt home removal, cauterization, cryotherapy, or application of caustic herbal remedies",
            "Protect the area strictly from UV radiation with physical clothing and SPF 50+ sunscreen",
            "Document size and appearance with sequential standardized photographs"
        ],
        "urgent": "rapid enlargement over weeks, spontaneous bleeding without trauma, new ulceration, or darkening."
    }
}

DEFAULT_CARE = {
    "external": ["Localized alteration in skin pigmentation, surface texture, or elevation"],
    "internal": ["Mild discomfort, tenderness, or itching"],
    "care": [
        "Keep the skin area clean, dry, and protected from environmental friction",
        "Avoid picking, scratching, or applying aggressive harsh astringents",
        "Consult a qualified healthcare professional or dermatologist for comprehensive diagnosis"
    ],
    "urgent": "symptoms worsen rapidly, fever develops, or severe pain and spreading inflammation occur."
}


def get_care_protocol(condition_name: str) -> Dict[str, Any]:
    """
    Retrieves the clinical care protocol for the specified condition.
    """
    for k in CARE_PROTOCOLS:
        if k.lower() in condition_name.lower() or condition_name.lower() in k.lower():
            return CARE_PROTOCOLS[k]
    return DEFAULT_CARE
