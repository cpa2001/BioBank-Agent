"""Biomarker feature groups and derived feature engineering.

Default field IDs target UK Biobank. Other biobanks should provide
their own feature group mappings via configuration.

The BIOMARKER_GROUPS dict is the canonical access point used by skills.
"""

from __future__ import annotations

# ── Default biomarker groups (UK Biobank field IDs) ────────────
# These serve as defaults. Other biobanks override via BankConfig.

BLOOD_BIOCHEMISTRY = {
    "30600": "Albumin",
    "30610": "Alkaline phosphatase",
    "30620": "Alanine aminotransferase",
    "30630": "Apolipoprotein A",
    "30640": "Apolipoprotein B",
    "30650": "Aspartate aminotransferase",
    "30660": "Direct bilirubin",
    "30670": "Urea",
    "30680": "Calcium",
    "30690": "Cholesterol",
    "30700": "Creatinine",
    "30710": "C-reactive protein",
    "30720": "Cystatin C",
    "30730": "Gamma glutamyltransferase",
    "30740": "Glucose",
    "30750": "Glycated haemoglobin (HbA1c)",
    "30760": "HDL cholesterol",
    "30770": "IGF-1",
    "30780": "LDL direct",
    "30790": "Lipoprotein A",
    "30800": "Oestradiol",
    "30810": "Phosphate",
    "30830": "SHBG",
    "30840": "Total bilirubin",
    "30850": "Testosterone",
    "30860": "Total protein",
    "30870": "Triglycerides",
    "30880": "Urate",
    "30890": "Vitamin D",
}

BLOOD_COUNT = {
    "30000": "White blood cell (leukocyte) count",
    "30010": "Red blood cell (erythrocyte) count",
    "30020": "Haemoglobin concentration",
    "30030": "Haematocrit percentage",
    "30040": "Mean corpuscular volume",
    "30050": "Mean corpuscular haemoglobin",
    "30060": "Mean corpuscular haemoglobin concentration",
    "30070": "Red blood cell distribution width",
    "30080": "Platelet count",
    "30090": "Platelet crit",
    "30100": "Mean platelet volume",
    "30110": "Platelet distribution width",
    "30120": "Lymphocyte count",
    "30130": "Monocyte count",
    "30140": "Neutrophill count",
    "30150": "Eosinophill count",
    "30160": "Basophill count",
    "30170": "Nucleated red blood cell count",
    "30180": "Lymphocyte percentage",
    "30190": "Monocyte percentage",
    "30200": "Neutrophill percentage",
    "30210": "Eosinophill percentage",
    "30220": "Basophill percentage",
    "30230": "Nucleated red blood cell percentage",
    "30240": "Reticulocyte percentage",
    "30250": "Reticulocyte count",
    "30260": "Mean reticulocyte volume",
    "30270": "Mean sphered cell volume",
    "30280": "Immature reticulocyte fraction",
    "30290": "High light scatter reticulocyte percentage",
    "30300": "High light scatter reticulocyte count",
}

ANTHROPOMETRIC = {
    "48": "Waist circumference",
    "49": "Hip circumference",
    "50": "Standing height",
    "21001": "Body mass index (BMI)",
    "23104": "Body mass index (BMI) (impedance)",
}

BLOOD_PRESSURE = {
    "4079": "Diastolic blood pressure (automated)",
    "4080": "Systolic blood pressure (automated)",
    "93": "Systolic blood pressure (manual)",
    "94": "Diastolic blood pressure (manual)",
    "102": "Pulse rate (automated)",
}

DEMOGRAPHICS = {
    "31": "Sex",
    "34": "Year of birth",
    "52": "Month of birth",
    "21000": "Ethnic background",
    "21003": "Age when attended assessment centre",
    "21022": "Age at recruitment",
}

LIFESTYLE = {
    "1558": "Alcohol intake frequency",
    "20116": "Smoking status",
    "1160": "Sleep duration",
    "20161": "Pack years of smoking",
}

# All biomarkers combined (for default model training)
ALL_BIOMARKERS = {
    **BLOOD_BIOCHEMISTRY,
    **BLOOD_COUNT,
    **ANTHROPOMETRIC,
    **BLOOD_PRESSURE,
}

# All features including demographics and lifestyle
ALL_FEATURES = {
    **ALL_BIOMARKERS,
    **DEMOGRAPHICS,
    **LIFESTYLE,
}


def get_feature_group(group_name: str) -> dict[str, str]:
    """Return a feature group by name."""
    groups = {
        "biochemistry": BLOOD_BIOCHEMISTRY,
        "blood_count": BLOOD_COUNT,
        "anthropometric": ANTHROPOMETRIC,
        "blood_pressure": BLOOD_PRESSURE,
        "demographics": DEMOGRAPHICS,
        "lifestyle": LIFESTYLE,
        "all_biomarkers": ALL_BIOMARKERS,
        "all_features": ALL_FEATURES,
    }
    return groups.get(group_name, ALL_BIOMARKERS)


def rename_columns(df, catalog=None) -> dict[str, str]:
    """Create a mapping from field-instance.array to human-readable names."""
    rename_map = {}
    for col in df.columns:
        if col in ("eid", "label"):
            continue
        fid = col.split("-")[0]
        name = ALL_FEATURES.get(fid)
        if name is None and catalog:
            name = catalog.field_name(fid)
        if name:
            rename_map[col] = name
    return rename_map
