"""Tests for deep_research search robustness."""

from types import SimpleNamespace


def _ctx():
    return SimpleNamespace(settings=SimpleNamespace(biobank_name="UK Biobank"))


def test_search_literature_uses_fallback_queries(monkeypatch):
    from biobank_agent.skills.deep_research import _search_literature
    from biobank_agent.skills import web_search as ws_mod

    calls = {"n": 0}

    def fake_web_search(query, max_results=10, ctx=None):
        calls["n"] += 1
        # Primary two passes return empty; fallback then returns one result.
        if calls["n"] <= 2:
            return {"results": []}
        return {"results": [{"title": "AMI biomarker paper", "url": "https://example.org/p1", "snippet": "x"}]}

    monkeypatch.setattr(ws_mod, "web_search", fake_web_search)
    out = _search_literature(
        topic="very long noisy topic with many constraints and symbols",
        max_sources=5,
        ctx=_ctx(),
    )
    assert calls["n"] > 2
    assert len(out) == 1
    assert out[0]["title"] == "AMI biomarker paper"


def test_search_literature_deduplicates_without_url(monkeypatch):
    from biobank_agent.skills.deep_research import _search_literature
    from biobank_agent.skills import web_search as ws_mod

    def fake_web_search(query, max_results=10, ctx=None):
        return {
            "results": [
                {"title": "Paper A", "url": "", "snippet": "s1"},
                {"title": "Paper A", "url": "", "snippet": "s2"},
                {"title": "Paper B", "url": "", "snippet": "s3"},
            ],
        }

    monkeypatch.setattr(ws_mod, "web_search", fake_web_search)
    out = _search_literature(topic="acute myocardial infarction", max_sources=10, ctx=_ctx())
    titles = [x.get("title") for x in out]
    assert titles.count("Paper A") == 1
    assert titles.count("Paper B") == 1


def test_search_literature_uses_europe_pmc_when_sparse(monkeypatch):
    from biobank_agent.skills.deep_research import _search_literature
    from biobank_agent.skills import web_search as ws_mod
    from biobank_agent.skills import deep_research as dr_mod

    monkeypatch.setattr(ws_mod, "web_search", lambda query, max_results=10, ctx=None: {"results": []})
    monkeypatch.setattr(
        dr_mod,
        "_search_europe_pmc",
        lambda topic, max_results: [
            {"title": "Europe PMC AMI paper", "url": "https://pubmed.ncbi.nlm.nih.gov/1/", "snippet": "x"},
        ],
    )

    out = _search_literature(topic="acute myocardial infarction biomarkers", max_sources=8, ctx=_ctx())
    assert len(out) == 1
    assert out[0]["title"] == "Europe PMC AMI paper"
