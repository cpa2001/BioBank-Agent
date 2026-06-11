"""Tests for Action Graph implicit-knowledge memory layer."""

from pathlib import Path

from biobank_agent.memory import LongTermMemory


def test_action_graph_claim_evidence_roundtrip(tmp_path: Path):
    mem = LongTermMemory(tmp_path / "memory")
    mem.upsert_node("claim", "c1", payload={"text": "LDL is associated with I21"}, score=0.8)
    mem.upsert_node("result", "r1", payload={"n_cases": 500, "p_value": 0.001}, score=1.0)
    mem.link_nodes(
        src_type="claim",
        src_id="c1",
        dst_type="result",
        dst_id="r1",
        relation="supports",
        weight=0.9,
        evidence={"source": "execution"},
    )

    evidence = mem.retrieve_claim_evidence("c1", limit=5)
    assert len(evidence) == 1
    assert evidence[0]["node_type"] == "result"
    assert evidence[0]["relation"] == "supports"

    explain = mem.explain_claim("c1", limit=5)
    assert "c1" in explain
    assert "supports" in explain


def test_action_graph_search_nodes(tmp_path: Path):
    mem = LongTermMemory(tmp_path / "memory")
    mem.upsert_node("paper", "pmid123", payload={"title": "Cardiovascular risk and LDL"}, score=0.7)
    hits = mem.action_graph.search_nodes("cardiovascular", node_types=["paper"], limit=5)
    assert hits
    assert hits[0]["node_type"] == "paper"
