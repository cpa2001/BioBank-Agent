"""Evidence Lattice — claim-level provenance tracking.

Extends the ActionGraph (Tier 8 memory) with typed evidence nodes that track
the full provenance chain for every scientific claim the agent makes.

Design source:
    GPT deep research: "Evidence Lattice — 不要让 agent 直接记'结论句子'，
    要让它记'声明节点 + 证据节点 + 转换节点 + 反证节点 + 生效时间区间'"

Architecture:
    ClaimRecord contains:
      - The assertion (claim_text)
      - Supporting/refuting evidence nodes (EvidenceNode list)
      - Confidence (computed from evidence weights)
      - Temporal validity (when this claim was valid)
      - Status (provisional → supported → refuted → expired)
      - Link back to StudySpec that generated it

    Evidence flows in from:
      - literature_qa skill (PaperQA2 citations)
      - Skill execution results (statistical outputs)
      - Verification checks (formal verification)
      - User feedback

Usage:
    from biobank_agent.evidence import EvidenceLattice, EvidenceNode

    lattice = EvidenceLattice(storage_path)
    claim_id = lattice.record_claim("HbA1c > 48 predicts T2D with AUC 0.78")
    lattice.add_evidence(claim_id, EvidenceNode(
        evidence_type="statistical_result",
        source_ref="train_model_E11_session_abc123",
        confidence=0.85,
        direction="supports",
    ))
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Evidence Node ────────────────────────────────────────────────────────────


@dataclass
class EvidenceNode:
    """A single piece of evidence supporting or refuting a claim.

    Attributes:
        evidence_id: Unique identifier for this evidence node
        evidence_type: Type of evidence (paper, statistical_result, data_observation, expert_opinion)
        source_ref: Reference to the source (DOI, skill execution ID, query hash)
        source_hash: SHA-256 hash of source content at link time (immutability proof)
        passage: Relevant text snippet or description
        confidence: Strength of this evidence [0.0, 1.0]
        direction: Whether this evidence supports, refutes, or is neutral
        temporal_validity: ISO date range (valid_from, valid_until) — None means always valid
        linked_at: When this evidence was linked to the claim
        metadata: Additional key-value data (e.g., page number, model version)
    """
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    evidence_type: str = "unknown"  # paper, statistical_result, data_observation, expert_opinion
    source_ref: str = ""           # DOI, file path, execution ID
    source_hash: str = ""          # SHA-256 of source content
    passage: str = ""              # Relevant text
    confidence: float = 0.5        # Evidence strength [0, 1]
    direction: str = "supports"    # supports, refutes, neutral
    temporal_validity: Optional[tuple[str, str]] = None  # (valid_from, valid_until) ISO
    linked_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceNode":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_skill_result(
        cls,
        skill_name: str,
        result: dict[str, Any],
        claim_direction: str = "supports",
        confidence: float = 0.7,
    ) -> "EvidenceNode":
        """Create an evidence node from a skill execution result."""
        content = json.dumps(result, sort_keys=True, default=str)
        return cls(
            evidence_type="statistical_result",
            source_ref=f"skill:{skill_name}",
            source_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
            passage=f"Result from {skill_name}: {str(result)[:200]}",
            confidence=confidence,
            direction=claim_direction,
            metadata={"skill": skill_name},
        )

    @classmethod
    def from_paper(
        cls,
        doi: str,
        title: str,
        passage: str,
        confidence: float = 0.6,
        direction: str = "supports",
    ) -> "EvidenceNode":
        """Create an evidence node from a paper citation."""
        return cls(
            evidence_type="paper",
            source_ref=doi or title,
            source_hash=hashlib.sha256(passage.encode()).hexdigest()[:16],
            passage=passage[:500],
            confidence=confidence,
            direction=direction,
            metadata={"title": title, "doi": doi},
        )


# ── Claim Record ─────────────────────────────────────────────────────────────


@dataclass
class ClaimRecord:
    """A scientific claim with its evidence lattice.

    Tracks the full provenance of a claim through supporting and
    refuting evidence, computing confidence from the balance of evidence.

    Status lifecycle: provisional → supported → refuted → expired
    """
    claim_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    claim_text: str = ""
    study_spec_id: Optional[str] = None  # Links back to StudySpec.spec_hash()
    evidence: list[EvidenceNode] = field(default_factory=list)
    overall_confidence: float = 0.0
    status: str = "provisional"  # provisional, supported, refuted, expired
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_evidence(self, node: EvidenceNode) -> None:
        """Add an evidence node and recompute confidence."""
        self.evidence.append(node)
        self.overall_confidence = self.compute_confidence()
        self.updated_at = datetime.now().isoformat()
        self._update_status()

    def compute_confidence(self) -> float:
        """Compute overall confidence from the balance of evidence.

        Weighted average where:
        - Supporting evidence increases confidence
        - Refuting evidence decreases it
        - Neutral evidence doesn't change it

        Returns: float in [0, 1] representing confidence in the claim
        """
        if not self.evidence:
            return 0.0

        supports = [e for e in self.evidence if e.direction == "supports"]
        refutes = [e for e in self.evidence if e.direction == "refutes"]

        sup_score = sum(e.confidence for e in supports)
        ref_score = sum(e.confidence for e in refutes)
        total = sup_score + ref_score

        if total == 0:
            return 0.0

        return round(sup_score / total, 4)

    def _update_status(self) -> None:
        """Auto-update status based on evidence balance."""
        conf = self.overall_confidence
        if len(self.evidence) >= 3:
            if conf >= 0.75:
                self.status = "supported"
            elif conf <= 0.25:
                self.status = "refuted"
            else:
                self.status = "provisional"

    @property
    def n_supporting(self) -> int:
        return sum(1 for e in self.evidence if e.direction == "supports")

    @property
    def n_refuting(self) -> int:
        return sum(1 for e in self.evidence if e.direction == "refutes")

    @property
    def n_neutral(self) -> int:
        return sum(1 for e in self.evidence if e.direction == "neutral")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "study_spec_id": self.study_spec_id,
            "evidence": [e.to_dict() for e in self.evidence],
            "overall_confidence": self.overall_confidence,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaimRecord":
        """Deserialize from dictionary."""
        evidence = [EvidenceNode.from_dict(e) for e in data.get("evidence", [])]
        return cls(
            claim_id=data.get("claim_id", str(uuid.uuid4())[:12]),
            claim_text=data.get("claim_text", ""),
            study_spec_id=data.get("study_spec_id"),
            evidence=evidence,
            overall_confidence=data.get("overall_confidence", 0.0),
            status=data.get("status", "provisional"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )


# ── Evidence Lattice (Storage & Query) ───────────────────────────────────────


class EvidenceLattice:
    """Persistent storage for claims and their evidence lattices.

    Stores claims as JSONL files, supports querying by status,
    confidence threshold, and study spec linkage.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the evidence lattice.

        Args:
            storage_path: Directory to persist claims. If None, operates in-memory only.
        """
        self.storage_path = storage_path
        self._claims: dict[str, ClaimRecord] = {}

        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def record_claim(
        self,
        claim_text: str,
        study_spec_id: Optional[str] = None,
        initial_evidence: Optional[list[EvidenceNode]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Record a new claim and return its ID.

        Args:
            claim_text: The scientific assertion
            study_spec_id: Link to the StudySpec that generated this claim
            initial_evidence: Optional initial evidence nodes
            metadata: Optional additional metadata

        Returns:
            The claim_id for future reference
        """
        claim = ClaimRecord(
            claim_text=claim_text,
            study_spec_id=study_spec_id,
            metadata=metadata or {},
        )
        if initial_evidence:
            for node in initial_evidence:
                claim.add_evidence(node)

        self._claims[claim.claim_id] = claim
        self._persist_claim(claim)
        return claim.claim_id

    def add_evidence(self, claim_id: str, evidence: EvidenceNode) -> bool:
        """Add evidence to an existing claim.

        Args:
            claim_id: The claim to add evidence to
            evidence: The evidence node to add

        Returns:
            True if successful, False if claim not found
        """
        if claim_id not in self._claims:
            logger.warning("Claim %s not found in lattice", claim_id)
            return False

        self._claims[claim_id].add_evidence(evidence)
        self._persist_claim(self._claims[claim_id])
        return True

    def get_claim(self, claim_id: str) -> Optional[ClaimRecord]:
        """Get a claim by ID."""
        return self._claims.get(claim_id)

    def get_claims_by_status(self, status: str) -> list[ClaimRecord]:
        """Get all claims with a given status."""
        return [c for c in self._claims.values() if c.status == status]

    def get_claims_by_spec(self, study_spec_id: str) -> list[ClaimRecord]:
        """Get all claims linked to a StudySpec."""
        return [c for c in self._claims.values() if c.study_spec_id == study_spec_id]

    def get_supported_claims(self, min_confidence: float = 0.7) -> list[ClaimRecord]:
        """Get claims that are well-supported by evidence."""
        return [c for c in self._claims.values()
                if c.overall_confidence >= min_confidence and len(c.evidence) >= 2]

    def get_contested_claims(self) -> list[ClaimRecord]:
        """Get claims with both supporting and refuting evidence."""
        return [c for c in self._claims.values()
                if c.n_supporting > 0 and c.n_refuting > 0]

    @property
    def n_claims(self) -> int:
        return len(self._claims)

    @property
    def n_evidence_total(self) -> int:
        return sum(len(c.evidence) for c in self._claims.values())

    def stats(self) -> dict[str, Any]:
        """Summary statistics of the evidence lattice."""
        status_counts = {}
        for claim in self._claims.values():
            status_counts[claim.status] = status_counts.get(claim.status, 0) + 1

        return {
            "n_claims": self.n_claims,
            "n_evidence_total": self.n_evidence_total,
            "status_distribution": status_counts,
            "avg_confidence": (
                sum(c.overall_confidence for c in self._claims.values()) / len(self._claims)
                if self._claims else 0.0
            ),
            "n_contested": len(self.get_contested_claims()),
        }

    def export_json(self) -> list[dict[str, Any]]:
        """Export entire lattice as JSON-serializable list."""
        return [c.to_dict() for c in self._claims.values()]

    def _persist_claim(self, claim: ClaimRecord) -> None:
        """Persist all claims to disk (atomic rewrite).

        Writes to a temp file and atomically renames to prevent
        corruption under concurrent access.
        """
        if not self.storage_path:
            return
        try:
            file = self.storage_path / "claims.jsonl"
            tmp = file.with_suffix(".jsonl.tmp")
            with open(tmp, "w") as f:
                for c in list(self._claims.values()):
                    f.write(json.dumps(c.to_dict(), default=str) + "\n")
            tmp.replace(file)
        except Exception as e:
            logger.warning("Failed to persist claim %s: %s", claim.claim_id, e)

    def _load_from_disk(self) -> None:
        """Load claims from disk on initialization."""
        if not self.storage_path:
            return
        file = self.storage_path / "claims.jsonl"
        if not file.exists():
            return
        try:
            latest_claims: dict[str, ClaimRecord] = {}
            for line in file.read_text().strip().split("\n"):
                if line:
                    data = json.loads(line)
                    claim = ClaimRecord.from_dict(data)
                    latest_claims[claim.claim_id] = claim  # Later entries override earlier
            self._claims = latest_claims
            logger.debug("Loaded %d claims from %s", len(self._claims), file)
        except Exception as e:
            logger.warning("Failed to load claims from %s: %s", file, e)
