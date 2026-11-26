"""Disposal / recycling knowledge base (bilingual), synced from shared/waste_knowledge.json."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "waste_knowledge.json"

# "recyclable" boolean stored with each prediction (for analytics). Compostable organic waste and
# special-handling hazardous waste are not counted as recyclable; e-waste via authorised recyclers is.
RECYCLABLE_TRUE = {"yes", "conditional", "authorized"}

_LIST_FIELDS = ("segregation", "disposal", "recycling", "safety", "common_mistakes", "environment")


@lru_cache
def knowledge() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def _as_list(value) -> list[str]:
    """Normalise a guidance field to a list of strings.

    `list("text")` would split a plain string into characters (one bullet per letter in the UI),
    so a string becomes a one-item list.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(x) for x in value if str(x).strip()]


def _bl(value) -> dict:
    value = value or {}
    return {"en": _as_list(value.get("en")), "hi": _as_list(value.get("hi"))}


def guidance_for(category: str) -> dict:
    kb = knowledge()
    c = kb["categories"][category]
    out = {
        "name": c["name"], "short": c["short"], "bin": c["bin"], "bin_label": kb["bins"][c["bin"]],
        "recyclable": c["recyclable"], "recyclable_label": kb["recyclable_labels"][c["recyclable"]],
        "color": c["color"], "icon": c["icon"],
    }
    for f in _LIST_FIELDS:
        out[f] = _bl(c.get(f))
    return out


def item_label(item: str | None) -> dict | None:
    if not item:
        return None
    return knowledge()["items"].get(item)


def is_recyclable(category: str) -> bool:
    return knowledge()["categories"][category]["recyclable"] in RECYCLABLE_TRUE


NOTICES = {
    "hazardous_scope": {
        "severity": "danger",
        "message": {
            "en": "Detection limitation: the model's Hazardous class was trained only on aerosol cans. Paint, "
                  "chemicals, pesticides, medicines, thermometers and CFL bulbs are not reliably detected. "
                  "Check the Waste Guide and never mix hazardous items with regular waste.",
            "hi": "पहचान की सीमा: मॉडल की 'खतरनाक' श्रेणी केवल एरोसोल कैन की तस्वीरों पर प्रशिक्षित है। पेंट, रसायन, "
                  "कीटनाशक, दवाइयाँ, थर्मामीटर और CFL बल्ब विश्वसनीय रूप से नहीं पहचाने जाते। अपशिष्ट गाइड देखें और "
                  "खतरनाक वस्तुओं को कभी भी सामान्य कचरे में न मिलाएँ।",
        },
    },
    "ewaste_scope": {
        "severity": "info",
        "message": {
            "en": "E-waste detection covers batteries and common devices. Chargers, cables and circuit boards were "
                  "not in the training data. All e-waste should go to an authorised e-waste collector.",
            "hi": "ई-कचरा पहचान बैटरी और सामान्य उपकरणों को कवर करती है। चार्जर, केबल और सर्किट बोर्ड प्रशिक्षण डेटा में "
                  "नहीं थे। सारा ई-कचरा अधिकृत ई-कचरा संग्रहकर्ता को दें।",
        },
    },
    "uncertain": {
        "severity": "warning",
        "message": {
            "en": "The model is not confident about this image. Treat the result as a suggestion. Try a clearer photo "
                  "of a single item on a plain background, or use the Waste Guide.",
            "hi": "मॉडल इस तस्वीर को लेकर आश्वस्त नहीं है। परिणाम को केवल सुझाव मानें। साफ़ पृष्ठभूमि पर एक ही वस्तु की "
                  "स्पष्ट तस्वीर लें, या अपशिष्ट गाइड का उपयोग करें।",
        },
    },
    "battery_safety": {
        "severity": "danger",
        "message": {
            "en": "Batteries can catch fire or leak. Never put them in household bins. Tape the terminals of lithium "
                  "batteries and hand them to an e-waste or battery collection point.",
            "hi": "बैटरियों में आग लग सकती है या रिसाव हो सकता है। इन्हें कभी घरेलू डिब्बे में न डालें। लिथियम बैटरी के "
                  "टर्मिनल पर टेप लगाएँ और ई-कचरा या बैटरी संग्रह केंद्र को दें।",
        },
    },
}


def notices_for(category: str, is_confident: bool, top_categories: list[str], item: str | None) -> list[dict]:
    codes: list[str] = []
    if not is_confident:
        codes.append("uncertain")
    # Show the hazardous limitation whenever hazardous is the answer or a close alternative.
    if category == "hazardous" or "hazardous" in top_categories[:2]:
        codes.append("hazardous_scope")
    if category == "e_waste":
        codes.append("ewaste_scope")
    if item == "battery":
        codes.append("battery_safety")
    return [{"code": c, **NOTICES[c]} for c in codes]
