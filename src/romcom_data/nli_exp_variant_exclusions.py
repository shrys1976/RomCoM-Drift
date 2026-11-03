"""Versioned EXP-02 variant exclusions (does not modify Dakshina inventory TSV)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from romcom_data.nli_exp_paths import NLI_INVENTORIES
from romcom_data.nli_exp_romanize import NliLexicon
from romcom_data.text_utils import nfc

EXCLUSIONS_JSON = NLI_INVENTORIES / "dakshina_hi_exp02_variant_exclusions.json"
AUDIT_SOURCE = "owner_followup_audit_2026-11-03"

# (native, variant) -> classification, reason, optional audit_evidence example_ids
_WRONG_WORD: dict[tuple[str, str], dict[str, Any]] = {
    ("कुल", "kulmilakar"): {
        "classification": "wrong_word_substitution",
        "reason": "Adds lexical material (altogether), not spelling of total/kul",
        "audit_evidence": ["train-002070"],
    },
    ("एक्सेस", "excess"): {
        "classification": "wrong_word_substitution",
        "reason": "English excess vs access",
        "audit_evidence": ["train-005426"],
    },
    ("वार", "warpalatwar"): {
        "classification": "wrong_word_substitution",
        "reason": "Introduces counterattack compound, not spelling of strike/war",
        "audit_evidence": ["train-004766"],
    },
    ("स्टारर", "starter"): {
        "classification": "wrong_word_substitution",
        "reason": "Different English word",
        "audit_evidence": [],
    },
    ("क्षेत्र", "kshem"): {
        "classification": "wrong_word_substitution",
        "reason": "Different Hindi lemma (welfare vs region/zone)",
        "audit_evidence": ["train-000620"],
    },
    ("इनाम", "imaan"): {
        "classification": "wrong_word_substitution",
        "reason": "Reward vs faith (imaan)",
        "audit_evidence": ["train-011479"],
    },
    ("कृतज्ञता", "kritagya"): {
        "classification": "wrong_word_substitution",
        "reason": "Truncated/gratitude vs grateful",
        "audit_evidence": [],
    },
    ("हैनकॉक", "hongkong"): {
        "classification": "wrong_word_substitution",
        "reason": "Proper noun Hancock vs Hong Kong",
        "audit_evidence": ["train-001224"],
    },
    ("एयरोस्पेस", "aerosports"): {
        "classification": "wrong_word_substitution",
        "reason": "Aerospace vs aerosports",
        "audit_evidence": [],
    },
    ("ट्रैक", "taker"): {
        "classification": "wrong_word_substitution",
        "reason": "Track vs taker in English loan context",
        "audit_evidence": ["train-006194"],
    },
    ("मोहम्मद", "moihabbat"): {
        "classification": "wrong_word_substitution",
        "reason": "Muhammad vs love (mohabbat)",
        "audit_evidence": ["train-000229"],
    },
    ("दब", "bad"): {
        "classification": "wrong_word_substitution",
        "reason": "Press/sink (dab) vs English bad",
        "audit_evidence": ["train-002320"],
    },
    ("बॉर्डर", "borer"): {
        "classification": "wrong_word_substitution",
        "reason": "Border vs borer",
        "audit_evidence": [],
    },
    ("व्हाट", "shat"): {
        "classification": "wrong_word_substitution",
        "reason": "What loan vs unrelated string",
        "audit_evidence": [],
    },
    ("फ्रीजिंग", "fridging"): {
        "classification": "wrong_word_substitution",
        "reason": "Freezing vs fridging",
        "audit_evidence": [],
    },
    ("दी", "the"): {
        "classification": "wrong_word_substitution",
        "reason": "Gave (di) vs English the",
        "audit_evidence": ["train-000543"],
    },
    ("आय", "aaya"): {
        "classification": "wrong_word_substitution",
        "reason": "Income (aay) vs came (aaya)",
        "audit_evidence": ["train-000140"],
    },
    ("दिलचस्प", "dilchaspi"): {
        "classification": "wrong_word_substitution",
        "reason": "Adjective interesting vs noun form",
        "audit_evidence": ["train-000368"],
    },
    ("रैट", "rate"): {
        "classification": "wrong_word_substitution",
        "reason": "Rat vs rate",
        "audit_evidence": [],
    },
    ("साइट", "sight"): {
        "classification": "wrong_word_substitution",
        "reason": "Site vs sight",
        "audit_evidence": [],
    },
    ("सेवन", "seven"): {
        "classification": "wrong_word_substitution",
        "reason": "Consumption vs seven",
        "audit_evidence": [],
    },
    ("एंटीसेप्टिक", "anticipate"): {
        "classification": "wrong_word_substitution",
        "reason": "Antiseptic vs anticipate",
        "audit_evidence": [],
    },
    ("दृग", "hug"): {
        "classification": "wrong_word_substitution",
        "reason": "Eye/drug sense vs hug",
        "audit_evidence": [],
    },
    ("वेलेस", "balance"): {
        "classification": "wrong_word_substitution",
        "reason": "Veles vs balance",
        "audit_evidence": [],
    },
    ("सायंस", "science"): {
        "classification": "wrong_word_substitution",
        "reason": "Hindi science loan vs English science spelling variant only if wrong lemma",
        "audit_evidence": [],
    },
    ("सोशलिज्म", "socialization"): {
        "classification": "wrong_word_substitution",
        "reason": "Socialism vs socialization",
        "audit_evidence": [],
    },
}

_GARBLED: dict[tuple[str, str], dict[str, Any]] = {
    ("अर्थात्", "arthart"): {"classification": "garbled_spelling", "reason": "Corrupted arthath/arthat", "audit_evidence": []},
    ("अर्थात", "arthart"): {"classification": "garbled_spelling", "reason": "Corrupted arthat", "audit_evidence": []},
    ("बोस्निया", "bostriyaa"): {"classification": "garbled_spelling", "reason": "Bosnia garble", "audit_evidence": []},
    ("बोस्निया", "bostriya"): {"classification": "garbled_spelling", "reason": "Bosnia garble", "audit_evidence": []},
    ("होस्नी", "hostri"): {"classification": "garbled_spelling", "reason": "Hosanna garble", "audit_evidence": []},
    ("तोड़फोड़", "tormor"): {"classification": "garbled_spelling", "reason": "Destruction word garble", "audit_evidence": []},
    ("सप्ताह", "satpaah"): {"classification": "garbled_spelling", "reason": "Week spelling corruption", "audit_evidence": ["train-000319"]},
    ("पश्चिम", "pakshim"): {"classification": "garbled_spelling", "reason": "West paschim garble", "audit_evidence": []},
    ("वैकल्पिक", "vaikplick"): {"classification": "garbled_spelling", "reason": "Optional vaikalpik garble", "audit_evidence": []},
    ("निर्दिष्ट", "nidrisht"): {"classification": "garbled_spelling", "reason": "Specified nirdisht garble", "audit_evidence": []},
    ("वक्र", "varka"): {"classification": "garbled_spelling", "reason": "Curve vakra garble", "audit_evidence": []},
    ("राष्ट्रों", "rashrton"): {"classification": "garbled_spelling", "reason": "Nations rashtron garble", "audit_evidence": []},
    ("शैक्षिक", "sheshik"): {"classification": "garbled_spelling", "reason": "Educational shaikshik garble", "audit_evidence": []},
    ("पुनर्स्थापित", "punstharpit"): {"classification": "garbled_spelling", "reason": "Restored punarsthapit garble", "audit_evidence": []},
    ("महत्वाकांक्षी", "mahtyakanshi"): {"classification": "garbled_spelling", "reason": "Ambitious mahatvakankshi garble", "audit_evidence": []},
    ("अंतर्देशीय", "antdreshiya"): {"classification": "garbled_spelling", "reason": "Domestic antardeshiy garble", "audit_evidence": []},
    ("मूर्ति", "mutri"): {"classification": "garbled_spelling", "reason": "Idol murti garble", "audit_evidence": []},
    ("चुटकुले", "chukuley"): {"classification": "garbled_spelling", "reason": "Jokes chutkule garble", "audit_evidence": []},
    ("महत्वाकांक्षा", "mahtyakansha"): {"classification": "garbled_spelling", "reason": "Ambition mahatvakanksha garble", "audit_evidence": []},
    ("राष्ट्रपतियों", "rashpatiyon"): {"classification": "garbled_spelling", "reason": "Presidents rashtrapatiyon garble", "audit_evidence": []},
    ("स्वीकार्यता", "sweekaryakarta"): {"classification": "garbled_spelling", "reason": "Acceptability swikaryata garble", "audit_evidence": []},
    ("सप्ताहों", "satpaahon"): {"classification": "garbled_spelling", "reason": "Weeks saptahon garble", "audit_evidence": []},
    ("वर्धित", "vadhirt"): {"classification": "garbled_spelling", "reason": "Increased vardhit garble", "audit_evidence": []},
    ("शुक्र", "surk"): {"classification": "garbled_spelling", "reason": "Venus shukra garble", "audit_evidence": []},
    ("आसन्न", "asatra"): {"classification": "garbled_spelling", "reason": "Imminent aasann garble", "audit_evidence": []},
    ("शिक्षिका", "shishika"): {"classification": "garbled_spelling", "reason": "Teacher shikshika garble", "audit_evidence": []},
    ("कारखाना", "kakhana"): {"classification": "garbled_spelling", "reason": "Factory karkhana garble", "audit_evidence": []},
    ("पार्श्वनाथ", "pasharvnath"): {"classification": "garbled_spelling", "reason": "Parshvanath garble", "audit_evidence": []},
    ("भर्त्सना", "bhatsarna"): {"classification": "garbled_spelling", "reason": "Rebuke bhartsana garble", "audit_evidence": []},
    ("वर्साय", "vasariya"): {"classification": "garbled_spelling", "reason": "Versailles varsaaye garble", "audit_evidence": []},
    ("उत्सर्जक", "utsajark"): {"classification": "garbled_spelling", "reason": "Emitter utsarjak garble", "audit_evidence": []},
    ("टेराकोटा", "teratoka"): {"classification": "garbled_spelling", "reason": "Terracotta terakota garble", "audit_evidence": []},
    ("निवर्तमान", "nirvatamaan"): {"classification": "garbled_spelling", "reason": "Returning nivartman garble", "audit_evidence": []},
    ("शियाओं", "xiaon"): {"classification": "garbled_spelling", "reason": "Xiao garble", "audit_evidence": []},
    ("पोर्टल", "potral"): {"classification": "garbled_spelling", "reason": "Portal garble", "audit_evidence": []},
    ("एप्लाइड", "eplight"): {"classification": "garbled_spelling", "reason": "Applied garble", "audit_evidence": []},
    ("बीटल्स", "builts"): {"classification": "garbled_spelling", "reason": "Beatles garble", "audit_evidence": []},
    ("रेजिस्टेंस", "registres"): {"classification": "garbled_spelling", "reason": "Resistance garble", "audit_evidence": []},
    ("दक्षिण", "dakshid"): {"classification": "garbled_spelling", "reason": "South dakshin garble", "audit_evidence": []},
    ("मील", "meeley"): {"classification": "garbled_spelling", "reason": "Mile garble", "audit_evidence": []},
    ("मिलियन", "millium"): {"classification": "garbled_spelling", "reason": "Million garble", "audit_evidence": []},
    ("ट्रेंट", "trat"): {"classification": "garbled_spelling", "reason": "Trent garble", "audit_evidence": []},
}

_QUARANTINE: dict[tuple[str, str], dict[str, Any]] = {
    ("क्षण", "shan"): {"classification": "quarantine_uncertain", "reason": "Moment vs English Shan; pending owner resolution", "audit_evidence": []},
    ("क्षणों", "shanon"): {"classification": "quarantine_uncertain", "reason": "Moments vs shanon; pending", "audit_evidence": []},
    ("क्षमा", "shma"): {"classification": "quarantine_uncertain", "reason": "Forgiveness kshama vs shma; pending", "audit_evidence": []},
    ("बंगाल", "bangla"): {"classification": "quarantine_uncertain", "reason": "Bengal bangal vs bangla; pending", "audit_evidence": []},
    ("पंचर", "puncher"): {"classification": "quarantine_uncertain", "reason": "Puncture punchar vs puncher; pending", "audit_evidence": []},
}


def _entry(
    native: str,
    variant: str,
    meta: dict[str, Any],
    *,
    lexicon: NliLexicon,
) -> dict[str, Any]:
    key = nfc(native.strip())
    canon = lexicon.native_to_canonical.get(key, "").casefold()
    return {
        "native_word": key,
        "canonical": canon,
        "variant": variant.casefold(),
        "classification": meta["classification"],
        "reason": meta["reason"],
        "source": AUDIT_SOURCE,
        "audit_evidence": meta.get("audit_evidence") or [],
    }


def build_exclusion_document(lexicon: NliLexicon | None = None) -> dict[str, Any]:
    lex = lexicon or NliLexicon()
    excluded = [
        _entry(n, v, m, lexicon=lex)
        for d in (_WRONG_WORD, _GARBLED)
        for (n, v), m in d.items()
    ]
    quarantine = [_entry(n, v, m, lexicon=lex) for (n, v), m in _QUARANTINE.items()]
    return {
        "version": "1.0",
        "source": AUDIT_SOURCE,
        "note": "Exact native/canonical/variant triples excluded from EXP-02 selection only.",
        "excluded_mappings": excluded,
        "quarantine_mappings": quarantine,
    }


def load_exclusion_document(path: Path | None = None) -> dict[str, Any]:
    path = path or EXCLUSIONS_JSON
    return json.loads(path.read_text(encoding="utf-8"))


def excluded_variant_keys(path: Path | None = None) -> set[tuple[str, str]]:
    """Set of (native_word_nfc, variant_casefold) to drop from EXP-02 pools."""
    doc = load_exclusion_document(path)
    out: set[tuple[str, str]] = set()
    for row in doc.get("excluded_mappings", []):
        out.add((nfc(row["native_word"]), str(row["variant"]).casefold()))
    return out


def exclusions_sha256(path: Path | None = None) -> str:
    path = path or EXCLUSIONS_JSON
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_exclusion_file(path: Path | None = None) -> Path:
    path = path or EXCLUSIONS_JSON
    doc = build_exclusion_document()
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
