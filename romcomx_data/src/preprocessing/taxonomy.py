"""Indi-RomCoM Table 9 word-selection taxonomy (deterministic reconstruction).

Paper categories (Das et al. 2026, Appendix B, Table 9–10):
- CAT-0: never switched (numbers, names, structural markers)
- CAT-C: switched at 25/50/75 (task/domain evaluation terms)
- CAT-B: switched at 50/75 (discourse markers / connectives)
- CAT-D: switched at 50/75 (concrete thematic nouns)
- CAT-A: switched at 75 (function words)
- CAT-E: switched at 75 (clause-level predicates) — approximated, not LLM-rewritten

Intensity activation (Table 10):
- 25%: {C}
- 50%: {B, C, D}
- 75%: {A, B, C, D, E}
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INTENSITY_CATEGORIES: dict[str, frozenset[str]] = {
    "25": frozenset({"C"}),
    "50": frozenset({"B", "C", "D"}),
    "75": frozenset({"A", "B", "C", "D", "E"}),
}

# English -> Hindi native. Closed lexicon from Table 9 examples + standard bilingual CM lists.
CAT_A: dict[str, str] = {
    "the": "वह",
    "a": "एक",
    "an": "एक",
    "is": "है",
    "are": "हैं",
    "was": "था",
    "were": "थे",
    "be": "होना",
    "been": "रहा",
    "being": "होते",
    "am": "हूँ",
    "of": "का",
    "to": "को",
    "in": "में",
    "on": "पर",
    "for": "के",
    "with": "साथ",
    "by": "द्वारा",
    "from": "से",
    "at": "पर",
    "as": "जैसा",
    "this": "यह",
    "that": "वह",
    "these": "ये",
    "those": "वे",
    "it": "यह",
    "its": "इसका",
    "they": "वे",
    "them": "उन्हें",
    "their": "उनका",
    "we": "हम",
    "our": "हमारा",
    "i": "मैं",
    "you": "आप",
    "he": "वह",
    "she": "वह",
    "his": "उसका",
    "her": "उसकी",
    "my": "मेरा",
    "me": "मुझे",
}

CAT_B: dict[str, str] = {
    "but": "लेकिन",
    "however": "हालांकि",
    "therefore": "इसलिए",
    "hence": "अतः",
    "because": "क्योंकि",
    "although": "हालांकि",
    "though": "हालांकि",
    "moreover": "इसके",
    "furthermore": "आगे",
    "thus": "इस",
    "so": "इसलिए",
    "if": "अगर",
    "then": "तो",
    "and": "और",
    "or": "या",
    "while": "जबकि",
    "yet": "फिर",
    "nevertheless": "फिर",
    "instead": "बजाय",
    "meanwhile": "इसी",
    "otherwise": "अन्यथा",
    "unless": "जबतक",
    "whether": "चाहे",
    "since": "चूंकि",
}

CAT_C: dict[str, str] = {
    "entailment": "अनुमान",
    "contradiction": "विरोधाभास",
    "neutral": "तटस्थ",
    "paraphrase": "पुनर्कथन",
    "premise": "आधार",
    "hypothesis": "परिकल्पना",
    "sentence": "वाक्य",
    "label": "लेबल",
    "toxic": "विषैला",
    "toxicity": "विषाक्तता",
    "harmful": "हानिकारक",
    "safe": "सुरक्षित",
    "refusal": "अस्वीकृति",
    "classification": "वर्गीकरण",
    "quality": "गुणवत्ता",
    "analogy": "उपमा",
    "causation": "कारण",
}

# High-precision CAT-D content nouns (Table 9 examples + frequent NLI/news nouns).
CAT_D: dict[str, str] = {
    "police": "पुलिस",
    "person": "व्यक्ति",
    "people": "लोग",
    "man": "आदमी",
    "woman": "औरत",
    "child": "बच्चा",
    "government": "सरकार",
    "country": "देश",
    "city": "शहर",
    "village": "गाँव",
    "school": "स्कूल",
    "student": "छात्र",
    "teacher": "शिक्षक",
    "hospital": "अस्पताल",
    "doctor": "डॉक्टर",
    "company": "कंपनी",
    "market": "बाजार",
    "money": "पैसा",
    "work": "काम",
    "job": "नौकरी",
    "family": "परिवार",
    "house": "घर",
    "home": "घर",
    "water": "पानी",
    "food": "खाना",
    "time": "समय",
    "year": "वर्ष",
    "day": "दिन",
    "night": "रात",
    "world": "दुनिया",
    "life": "जीवन",
    "death": "मौत",
    "war": "युद्ध",
    "peace": "शांति",
    "law": "कानून",
    "court": "अदालत",
    "president": "राष्ट्रपति",
    "minister": "मंत्री",
    "party": "पार्टी",
    "election": "चुनाव",
    "news": "समाचार",
    "report": "रिपोर्ट",
    "problem": "समस्या",
    "question": "सवाल",
    "answer": "जवाब",
    "information": "जानकारी",
    "example": "उदाहरण",
    "result": "परिणाम",
    "study": "अध्ययन",
    "research": "शोध",
    "group": "समूह",
    "team": "टीम",
    "game": "खेल",
    "film": "फिल्म",
    "book": "किताब",
    "language": "भाषा",
    "word": "शब्द",
    "story": "कहानी",
    "history": "इतिहास",
    "science": "विज्ञान",
    "health": "स्वास्थ्य",
    "power": "शक्ति",
    "energy": "ऊर्जा",
    "car": "कार",
    "road": "सड़क",
    "train": "रेल",
    "airport": "हवाईअड्डा",
    "battery": "बैटरी",
}

CAT_E_CLAUSE_HINTS = frozenset(
    {
        "cannot",
        "can't",
        "refuse",
        "will not",
        "do not",
        "don't",
    }
)

NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
URL_RE = re.compile(r"https?://|www\.", re.I)
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
HANDLE_RE = re.compile(r"^[@#]\w+$")


@dataclass(frozen=True)
class TokenDecision:
    token: str
    category: str
    hindi_native: str | None
    switch: bool
    reason: str


def classify_english_token(token: str, *, sentence_initial: bool = False) -> str:
    raw = token
    low = token.casefold()
    if raw.isspace() or not re.search(r"[A-Za-z0-9]", raw):
        return "0"
    if NUMBER_RE.match(raw) or URL_RE.search(raw) or EMAIL_RE.search(raw) or HANDLE_RE.match(raw):
        return "0"
    if raw.isupper() and len(raw) >= 2:
        return "0"
    if raw[:1].isupper() and not sentence_initial and low not in CAT_A and low not in CAT_B:
        return "0"
    if low in CAT_C:
        return "C"
    if low in CAT_B:
        return "B"
    if low in CAT_A:
        return "A"
    if low in CAT_D:
        return "D"
    if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", raw):
        return "D"
    return "0"


def hindi_for_category(token: str, category: str) -> str | None:
    low = token.casefold()
    mapping = {"A": CAT_A, "B": CAT_B, "C": CAT_C, "D": CAT_D}
    table = mapping.get(category, {})
    return table.get(low)


def should_switch(category: str, intensity: str) -> bool:
    if category == "0":
        return False
    return category in INTENSITY_CATEGORIES[intensity]
