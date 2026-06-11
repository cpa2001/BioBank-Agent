"""ICD10 code utilities — tree, lookup, chapter mapping."""

from __future__ import annotations

# ── ICD10 Chapter mapping ───────────────────────────────────

ICD10_CHAPTERS = {
    "A": ("I", "Certain infectious and parasitic diseases"),
    "B": ("I", "Certain infectious and parasitic diseases"),
    "C": ("II", "Neoplasms"),
    "D": ("II/III", "Neoplasms / Blood diseases"),
    "E": ("IV", "Endocrine, nutritional and metabolic diseases"),
    "F": ("V", "Mental and behavioural disorders"),
    "G": ("VI", "Diseases of the nervous system"),
    "H": ("VII/VIII", "Eye and ear diseases"),
    "I": ("IX", "Diseases of the circulatory system"),
    "J": ("X", "Diseases of the respiratory system"),
    "K": ("XI", "Diseases of the digestive system"),
    "L": ("XII", "Diseases of the skin"),
    "M": ("XIII", "Diseases of the musculoskeletal system"),
    "N": ("XIV", "Diseases of the genitourinary system"),
    "O": ("XV", "Pregnancy, childbirth and the puerperium"),
    "P": ("XVI", "Certain conditions originating in the perinatal period"),
    "Q": ("XVII", "Congenital malformations"),
    "R": ("XVIII", "Symptoms, signs and abnormal clinical findings"),
    "S": ("XIX", "Injury, poisoning"),
    "T": ("XIX", "Injury, poisoning"),
    "V": ("XX", "External causes of morbidity"),
    "W": ("XX", "External causes of morbidity"),
    "X": ("XX", "External causes of morbidity"),
    "Y": ("XX", "External causes of morbidity"),
    "Z": ("XXI", "Factors influencing health status"),
}

# Common diseases for quick lookup
COMMON_DISEASES = {
    "E11": "Type 2 diabetes mellitus",
    "I10": "Essential (primary) hypertension",
    "I25": "Chronic ischaemic heart disease",
    "I21": "Acute myocardial infarction",
    "E78": "Disorders of lipoprotein metabolism",
    "J45": "Asthma",
    "K80": "Cholelithiasis (gallstones)",
    "G47": "Sleep disorders",
    "E03": "Other hypothyroidism",
    "N20": "Calculus of kidney and ureter",
    "M54": "Dorsalgia (back pain)",
    "J44": "Chronic obstructive pulmonary disease",
    "F32": "Depressive episode",
    "E66": "Obesity",
    "N18": "Chronic kidney disease",
    "C50": "Malignant neoplasm of breast",
    "C61": "Malignant neoplasm of prostate",
    "C34": "Malignant neoplasm of bronchus and lung",
    "G30": "Alzheimer disease",
    "I48": "Atrial fibrillation and flutter",
}


def icd10_name(code: str) -> str:
    """Return human-readable name for an ICD10 code."""
    # Try exact match first
    if code in COMMON_DISEASES:
        return COMMON_DISEASES[code]
    # Try 3-char prefix
    prefix = code[:3]
    if prefix in COMMON_DISEASES:
        return COMMON_DISEASES[prefix]
    return code


def icd10_chapter(code: str) -> tuple[str, str]:
    """Return (chapter_number, chapter_name) for an ICD10 code."""
    if not code:
        return ("?", "Unknown")
    first_letter = code[0].upper()
    return ICD10_CHAPTERS.get(first_letter, ("?", "Unknown"))


def is_same_chapter(code1: str, code2: str) -> bool:
    """Check if two ICD10 codes belong to the same chapter."""
    return icd10_chapter(code1)[0] == icd10_chapter(code2)[0]
