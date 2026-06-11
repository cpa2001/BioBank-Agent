"""Cross-cohort phenotype harmonisation primitives.

The first version is schema-first: it creates auditable endpoint mappings before
raw HPP/CKB data are available, then writes those mappings into Action Graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..banks import load_bank


@dataclass(frozen=True)
class BankPhenotypeMapping:
    """One cohort-specific representation of a shared phenotype concept."""

    bank_id: str
    display_name: str
    phenotype_type: str
    code_system: str
    codes: list[str]
    field_ids: list[str] = field(default_factory=list)
    unit: str = ""
    source: str = ""
    case_definition: str = ""
    time_anchor: str = "baseline_or_first_observed"
    caveats: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HarmonizedPhenotype:
    """A cross-cohort endpoint/feature definition with explicit drift risks."""

    concept_id: str
    label: str
    phenotype_type: str
    mappings: list[BankPhenotypeMapping]
    harmonisation_status: str
    drift_risks: list[str] = field(default_factory=list)
    recommended_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "label": self.label,
            "phenotype_type": self.phenotype_type,
            "mappings": [asdict(m) for m in self.mappings],
            "harmonisation_status": self.harmonisation_status,
            "drift_risks": list(self.drift_risks),
            "recommended_checks": list(self.recommended_checks),
        }


_DISEASE_CONCEPTS: dict[str, dict[str, Any]] = {
    "i21": {
        "concept_id": "acute_myocardial_infarction",
        "label": "Acute myocardial infarction",
        "codes": ["I21"],
        "checks": ["incident/prevalent split", "death-registry linkage", "prior CVD exclusion window"],
    },
    "e11": {
        "concept_id": "type_2_diabetes",
        "label": "Type 2 diabetes mellitus",
        "codes": ["E11"],
        "checks": ["medication evidence", "HbA1c/glucose triangulation", "prevalent diabetes exclusion"],
    },
    "i10": {
        "concept_id": "essential_hypertension",
        "label": "Essential hypertension",
        "codes": ["I10"],
        "checks": ["blood-pressure measurement protocol", "antihypertensive medication evidence"],
    },
    "n18": {
        "concept_id": "chronic_kidney_disease",
        "label": "Chronic kidney disease",
        "codes": ["N18"],
        "checks": ["eGFR equation", "albuminuria availability", "incident CKD timing"],
    },
    "c34": {
        "concept_id": "lung_cancer",
        "label": "Malignant neoplasm of bronchus and lung",
        "codes": ["C34"],
        "checks": ["cancer registry linkage", "smoking adjustment", "histology availability"],
    },
}

_BIOMARKER_CONCEPTS: dict[str, dict[str, Any]] = {
    "ldl": {
        "concept_id": "ldl_cholesterol",
        "label": "LDL cholesterol",
        "unit": "mmol/L",
        "fields": {"ukb": ["30780"], "hpp": ["hpp_ldl"], "ckb": ["ckb_ldl"]},
    },
    "hba1c": {
        "concept_id": "hba1c",
        "label": "Glycated haemoglobin (HbA1c)",
        "unit": "mmol/mol",
        "fields": {"ukb": ["30750"], "hpp": ["hpp_hba1c"], "ckb": ["ckb_hba1c"]},
    },
    "sbp": {
        "concept_id": "systolic_blood_pressure",
        "label": "Systolic blood pressure",
        "unit": "mmHg",
        "fields": {"ukb": ["4080", "93"], "hpp": ["hpp_sbp"], "ckb": ["ckb_sbp"]},
    },
    "bmi": {
        "concept_id": "body_mass_index",
        "label": "Body mass index",
        "unit": "kg/m^2",
        "fields": {"ukb": ["21001"], "hpp": ["hpp_bmi"], "ckb": ["ckb_bmi"]},
    },
    "crp": {
        "concept_id": "c_reactive_protein",
        "label": "C-reactive protein",
        "unit": "mg/L",
        "fields": {"ukb": ["30710"], "hpp": ["hpp_crp"], "ckb": ["ckb_crp"]},
    },
}


def _normalize_concept(concept: str) -> str:
    key = (concept or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "myocardial_infarction": "i21",
        "acute_mi": "i21",
        "mi": "i21",
        "type_2_diabetes": "e11",
        "t2d": "e11",
        "diabetes": "e11",
        "hypertension": "i10",
        "ckd": "n18",
        "chronic_kidney_disease": "n18",
        "lung_cancer": "c34",
        "ldl_cholesterol": "ldl",
        "systolic_blood_pressure": "sbp",
        "body_mass_index": "bmi",
        "c_reactive_protein": "crp",
    }
    return aliases.get(key, key)


def harmonize_phenotype(
    concept: str,
    banks: list[str] | None = None,
    phenotype_type: str = "disease",
) -> HarmonizedPhenotype:
    """Build a cross-cohort phenotype mapping for UKB/HPP/CKB style data."""

    banks = banks or ["ukb", "hpp", "ckb"]
    key = _normalize_concept(concept)
    phenotype_type = phenotype_type.lower().strip()

    if phenotype_type == "biomarker" or key in _BIOMARKER_CONCEPTS:
        spec = _BIOMARKER_CONCEPTS.get(key)
        if spec is None:
            raise ValueError(f"Unknown biomarker concept: {concept}")
        mappings = []
        for bank_id in banks:
            bank = load_bank(bank_id)
            field_ids = list(spec["fields"].get(bank_id, []))
            mappings.append(BankPhenotypeMapping(
                bank_id=bank_id,
                display_name=bank.display_name,
                phenotype_type="biomarker",
                code_system="field",
                codes=[],
                field_ids=field_ids,
                unit=spec.get("unit", ""),
                source="field_dictionary",
                case_definition=f"{spec['label']} measured at {bank.default_instance} / first available visit",
                caveats=[
                    "unit conversion required before pooled analysis",
                    "assay platform and fasting status may differ across cohorts",
                ],
            ))
        return HarmonizedPhenotype(
            concept_id=spec["concept_id"],
            label=spec["label"],
            phenotype_type="biomarker",
            mappings=mappings,
            harmonisation_status="READY" if all(m.field_ids for m in mappings) else "PARTIAL",
            drift_risks=["unit_shift", "assay_platform_shift", "missingness_shift"],
            recommended_checks=["unit harmonisation", "per-cohort missingness", "distribution shift"],
        )

    spec = _DISEASE_CONCEPTS.get(key)
    if spec is None:
        # Treat unknown ICD-like inputs as schema-valid but partial.
        spec = {
            "concept_id": key.lower(),
            "label": f"ICD-coded phenotype {concept}",
            "codes": [concept.upper()],
            "checks": ["manual code review", "case definition validation"],
        }

    mappings = []
    for bank_id in banks:
        bank = load_bank(bank_id)
        mappings.append(BankPhenotypeMapping(
            bank_id=bank_id,
            display_name=bank.display_name,
            phenotype_type="disease",
            code_system=bank.coding_system,
            codes=list(spec["codes"]),
            source="diagnoses/deaths linkage",
            case_definition=(
                f"participant has any {bank.coding_system} code prefix in {spec['codes']} "
                "from linked diagnoses or cause-of-death records"
            ),
            time_anchor="first diagnosis date when available; otherwise first observed record",
            caveats=[
                bank.caveats,
                "incident/prevalent status depends on available diagnosis dates",
            ],
        ))

    return HarmonizedPhenotype(
        concept_id=spec["concept_id"],
        label=spec["label"],
        phenotype_type="disease",
        mappings=mappings,
        harmonisation_status="READY",
        drift_risks=[
            "code_source_shift",
            "case_ascertainment_shift",
            "healthcare_system_shift",
            "follow_up_window_shift",
        ],
        recommended_checks=list(spec["checks"]) + [
            "minimum case count per cohort",
            "age/sex/ancestry adjustment",
            "measurement and registry coverage audit",
        ],
    )


def record_phenotype_to_action_graph(memory: Any, phenotype: HarmonizedPhenotype) -> None:
    """Persist phenotype mapping into Action Graph when memory is available."""

    if memory is None or not hasattr(memory, "upsert_node"):
        return
    endpoint_id = phenotype.concept_id
    memory.upsert_node(
        "phenotype",
        endpoint_id,
        payload=phenotype.to_dict(),
        score=1.0,
    )
    for mapping in phenotype.mappings:
        mapping_id = f"{endpoint_id}:{mapping.bank_id}"
        memory.upsert_node("mapping", mapping_id, payload=asdict(mapping), score=0.9)
        memory.link_nodes(
            "phenotype",
            endpoint_id,
            "mapping",
            mapping_id,
            relation="has_cohort_mapping",
            weight=0.9,
            evidence={"source": "phenotype_harmonisation"},
        )
        for field_id in mapping.field_ids:
            memory.upsert_node("field", f"{mapping.bank_id}:{field_id}", payload=asdict(mapping), score=0.8)
            memory.link_nodes(
                "mapping",
                mapping_id,
                "field",
                f"{mapping.bank_id}:{field_id}",
                relation="uses_field",
                weight=0.8,
            )
