"""Feature group compatibility tests for downstream skills."""


def test_biomarker_groups_canonical_export_exists():
    from biobank_agent.data.features import BIOMARKER_GROUPS

    assert "blood_biochemistry" in BIOMARKER_GROUPS
    assert "30740" in BIOMARKER_GROUPS["blood_biochemistry"]
    assert "anthropometric" in BIOMARKER_GROUPS
    assert "21001" in BIOMARKER_GROUPS["anthropometric"]
