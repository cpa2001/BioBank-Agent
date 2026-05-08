"""Tests for Evidence Lattice (Phase 7).

Tests evidence node creation, claim records, confidence computation,
persistence, and lattice queries.
"""

import json
import builtins
import pytest
from pathlib import Path
from biobank_agent.evidence import (
    EvidenceNode,
    ClaimRecord,
    EvidenceLattice,
)


@pytest.fixture
def lattice(tmp_path):
    """Evidence lattice with temp storage."""
    return EvidenceLattice(storage_path=tmp_path)


@pytest.fixture
def sample_evidence():
    """Sample evidence nodes for testing."""
    return [
        EvidenceNode(
            evidence_type="paper",
            source_ref="10.1038/s41588-024-01898-1",
            passage="HbA1c > 48 mmol/mol is diagnostic for T2D",
            confidence=0.9,
            direction="supports",
        ),
        EvidenceNode(
            evidence_type="statistical_result",
            source_ref="skill:train_model_E11",
            passage="Model AUC = 0.78 on test set",
            confidence=0.85,
            direction="supports",
        ),
        EvidenceNode(
            evidence_type="paper",
            source_ref="10.1234/contradicting-study",
            passage="HbA1c alone is insufficient for diagnosis",
            confidence=0.6,
            direction="refutes",
        ),
    ]


class TestEvidenceNode:
    """Test evidence node creation and serialization."""

    def test_default_creation(self):
        """Default evidence node should have valid defaults."""
        node = EvidenceNode()
        assert len(node.evidence_id) == 12
        assert node.evidence_type == "unknown"
        assert node.confidence == 0.5
        assert node.direction == "supports"

    def test_from_dict_roundtrip(self):
        """to_dict → from_dict should produce equivalent node."""
        node = EvidenceNode(
            evidence_type="paper",
            source_ref="10.1038/test",
            confidence=0.9,
            direction="supports",
            passage="Test passage",
        )
        data = node.to_dict()
        restored = EvidenceNode.from_dict(data)
        assert restored.evidence_type == "paper"
        assert restored.confidence == 0.9
        assert restored.passage == "Test passage"

    def test_from_skill_result(self):
        """Should create node from skill execution result."""
        result = {"auc": 0.82, "n_cases": 5000}
        node = EvidenceNode.from_skill_result("train_model", result)
        assert node.evidence_type == "statistical_result"
        assert node.source_ref == "skill:train_model"
        assert len(node.source_hash) == 16

    def test_from_paper(self):
        """Should create node from paper citation."""
        node = EvidenceNode.from_paper(
            doi="10.1038/test",
            title="Test Paper",
            passage="Key finding here",
            confidence=0.8,
        )
        assert node.evidence_type == "paper"
        assert node.metadata["doi"] == "10.1038/test"


class TestClaimRecord:
    """Test claim records and confidence computation."""

    def test_empty_claim(self):
        """Empty claim should have 0 confidence."""
        claim = ClaimRecord(claim_text="Test claim")
        assert claim.overall_confidence == 0.0
        assert claim.compute_confidence() == 0.0
        assert claim.status == "provisional"
        assert claim.n_supporting == 0

    def test_add_evidence(self, sample_evidence):
        """Adding evidence should update confidence."""
        claim = ClaimRecord(claim_text="HbA1c predicts T2D")
        claim.add_evidence(sample_evidence[0])
        assert claim.overall_confidence > 0
        assert claim.n_supporting == 1

    def test_confidence_with_supports_only(self):
        """All supporting evidence should give confidence = 1.0."""
        claim = ClaimRecord(claim_text="Strong claim")
        claim.add_evidence(EvidenceNode(confidence=0.9, direction="supports"))
        claim.add_evidence(EvidenceNode(confidence=0.8, direction="supports"))
        assert claim.compute_confidence() == 1.0

    def test_confidence_with_refutes(self, sample_evidence):
        """Mixed evidence should produce intermediate confidence."""
        claim = ClaimRecord(claim_text="Contested claim")
        for e in sample_evidence:
            claim.add_evidence(e)
        conf = claim.compute_confidence()
        # 0.9 + 0.85 supports vs 0.6 refutes → should be > 0.5
        assert 0.5 < conf < 1.0

    def test_status_auto_update_supported(self):
        """3+ evidence with high confidence should set 'supported'."""
        claim = ClaimRecord(claim_text="Well supported")
        for _ in range(4):
            claim.add_evidence(EvidenceNode(confidence=0.9, direction="supports"))
        assert claim.status == "supported"

    def test_status_auto_update_refuted(self):
        """3+ evidence mostly refuting should set 'refuted'."""
        claim = ClaimRecord(claim_text="Refuted claim")
        claim.add_evidence(EvidenceNode(confidence=0.1, direction="supports"))
        claim.add_evidence(EvidenceNode(confidence=0.9, direction="refutes"))
        claim.add_evidence(EvidenceNode(confidence=0.8, direction="refutes"))
        assert claim.status == "refuted"

    def test_n_neutral(self):
        """Should count neutral evidence correctly."""
        claim = ClaimRecord(claim_text="Test")
        claim.add_evidence(EvidenceNode(direction="neutral"))
        claim.add_evidence(EvidenceNode(direction="neutral"))
        assert claim.n_neutral == 2

    def test_to_dict_roundtrip(self, sample_evidence):
        """Serialization roundtrip should preserve data."""
        claim = ClaimRecord(claim_text="Roundtrip test", study_spec_id="abc123")
        for e in sample_evidence:
            claim.add_evidence(e)
        data = claim.to_dict()
        restored = ClaimRecord.from_dict(data)
        assert restored.claim_text == "Roundtrip test"
        assert restored.study_spec_id == "abc123"
        assert len(restored.evidence) == 3


class TestEvidenceLattice:
    """Test the lattice storage and query layer."""

    def test_record_claim(self, lattice):
        """Should record a claim and return its ID."""
        claim_id = lattice.record_claim("Test claim about diabetes", initial_evidence=[EvidenceNode()])
        assert len(claim_id) == 12
        assert lattice.n_claims == 1
        assert lattice.get_claim(claim_id).n_supporting == 1

    def test_add_evidence_to_claim(self, lattice):
        """Should add evidence to an existing claim."""
        claim_id = lattice.record_claim("HbA1c predicts T2D")
        success = lattice.add_evidence(claim_id, EvidenceNode(
            evidence_type="paper",
            source_ref="10.1038/test",
            confidence=0.8,
        ))
        assert success
        claim = lattice.get_claim(claim_id)
        assert len(claim.evidence) == 1

    def test_add_evidence_unknown_claim(self, lattice):
        """Should return False for unknown claim ID."""
        success = lattice.add_evidence("nonexistent", EvidenceNode())
        assert not success

    def test_get_claims_by_status(self, lattice):
        """Should filter claims by status."""
        lattice.record_claim("Provisional claim 1")
        lattice.record_claim("Provisional claim 2")
        claims = lattice.get_claims_by_status("provisional")
        assert len(claims) == 2

    def test_get_claims_by_spec(self, lattice):
        """Should filter claims by study spec ID."""
        lattice.record_claim("Claim A", study_spec_id="spec_001")
        lattice.record_claim("Claim B", study_spec_id="spec_002")
        lattice.record_claim("Claim C", study_spec_id="spec_001")
        claims = lattice.get_claims_by_spec("spec_001")
        assert len(claims) == 2

    def test_get_supported_claims(self, lattice):
        """Should return well-supported claims."""
        claim_id = lattice.record_claim("Strong claim")
        for _ in range(3):
            lattice.add_evidence(claim_id, EvidenceNode(confidence=0.9, direction="supports"))
        supported = lattice.get_supported_claims(min_confidence=0.7)
        assert len(supported) == 1

    def test_get_contested_claims(self, lattice):
        """Should return claims with both supporting and refuting evidence."""
        claim_id = lattice.record_claim("Contested")
        lattice.add_evidence(claim_id, EvidenceNode(confidence=0.8, direction="supports"))
        lattice.add_evidence(claim_id, EvidenceNode(confidence=0.6, direction="refutes"))
        contested = lattice.get_contested_claims()
        assert len(contested) == 1

    def test_persistence_to_disk(self, tmp_path):
        """Claims should persist to JSONL file."""
        lattice = EvidenceLattice(storage_path=tmp_path)
        lattice.record_claim("Persistent claim")
        file = tmp_path / "claims.jsonl"
        assert file.exists()
        lines = file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["claim_text"] == "Persistent claim"

    def test_reload_from_disk(self, tmp_path):
        """New lattice instance should load existing claims."""
        # Write some claims
        lat1 = EvidenceLattice(storage_path=tmp_path)
        lat1.record_claim("Claim 1")
        lat1.record_claim("Claim 2")
        # New instance should load
        lat2 = EvidenceLattice(storage_path=tmp_path)
        assert lat2.n_claims == 2

    def test_reload_skips_blank_lines(self, tmp_path):
        """Blank JSONL lines should be ignored during reload."""
        lat1 = EvidenceLattice(storage_path=tmp_path)
        lat1.record_claim("Claim 1")
        lat1.record_claim("Claim 2")
        file = tmp_path / "claims.jsonl"
        lines = file.read_text().splitlines()
        file.write_text(lines[0] + "\n\n" + lines[1] + "\n", encoding="utf-8")

        lat2 = EvidenceLattice(storage_path=tmp_path)

        assert lat2.n_claims == 2

    def test_stats(self, lattice, sample_evidence):
        """Stats should report correct counts."""
        claim_id = lattice.record_claim("Stats test")
        for e in sample_evidence:
            lattice.add_evidence(claim_id, e)
        stats = lattice.stats()
        assert stats["n_claims"] == 1
        assert stats["n_evidence_total"] == 3
        assert "provisional" in stats["status_distribution"] or "supported" in stats["status_distribution"]

    def test_in_memory_mode(self):
        """Should work without storage_path (in-memory only)."""
        lattice = EvidenceLattice(storage_path=None)
        claim_id = lattice.record_claim("Memory only claim")
        assert lattice.n_claims == 1
        assert lattice.get_claim(claim_id) is not None
        lattice._load_from_disk()

    def test_export_json(self, lattice, sample_evidence):
        """Should export full lattice as JSON."""
        claim_id = lattice.record_claim("Export test")
        for e in sample_evidence[:2]:
            lattice.add_evidence(claim_id, e)
        exported = lattice.export_json()
        assert len(exported) == 1
        assert len(exported[0]["evidence"]) == 2

    def test_persistence_and_load_failures_are_non_fatal(self, tmp_path, monkeypatch):
        lattice = EvidenceLattice(storage_path=tmp_path)
        claim_id = lattice.record_claim("Will fail to persist")

        real_open = builtins.open

        def failing_open(path, *args, **kwargs):
            if str(path).endswith(".tmp"):
                raise OSError("disk full")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", failing_open)
        assert lattice.add_evidence(claim_id, EvidenceNode()) is True

        (tmp_path / "claims.jsonl").write_text("{not-json\n", encoding="utf-8")
        loaded = EvidenceLattice(storage_path=tmp_path)
        assert loaded.n_claims == 0
