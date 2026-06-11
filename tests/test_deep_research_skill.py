"""Tests for the deep_research literature and biobank cross-reference skill."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

from biobank_agent.skills import deep_research as deep_mod


class FakeHTTPResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def test_search_europe_pmc_success_empty_and_error_paths(monkeypatch):
    def fake_get(url, timeout=20):
        assert "pageSize=2" in url
        return FakeHTTPResponse(
            {
                "resultList": {
                    "result": [
                        {
                            "title": "Diabetes biomarker study",
                            "doi": "10.1038/example",
                            "abstractText": "A long abstract",
                        },
                        {
                            "title": "PubMed only",
                            "pmid": "12345",
                            "authorString": "Smith et al.",
                        },
                        {"title": "", "doi": "10.bad/skip"},
                    ]
                }
            }
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=fake_get))
    results = deep_mod._search_europe_pmc("type 2 diabetes biomarkers", 2)

    assert results == [
        {"title": "Diabetes biomarker study", "url": "https://doi.org/10.1038/example", "snippet": "A long abstract"},
        {"title": "PubMed only", "url": "https://pubmed.ncbi.nlm.nih.gov/12345/", "snippet": "Smith et al."},
    ]
    assert deep_mod._search_europe_pmc("   ", 2) == []

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline"))))
    assert deep_mod._search_europe_pmc("topic", 2) == []

    real_import = builtins.__import__

    def no_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_httpx)
    assert deep_mod._search_europe_pmc("topic", 2) == []


def test_search_europe_pmc_skips_empty_title_and_uses_source_id(monkeypatch):
    def fake_get(url, timeout=20):
        return FakeHTTPResponse(
            {
                "resultList": {
                    "result": [
                        {"title": "", "id": "SKIP"},
                        {"title": "Europe PMC only", "id": "MED/123", "abstractText": "From PMC"},
                    ]
                }
            }
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=fake_get))

    assert deep_mod._search_europe_pmc("topic", 5) == [
        {"title": "Europe PMC only", "url": "https://europepmc.org/article/MED/123", "snippet": "From PMC"}
    ]


def test_search_literature_deduplicates_and_uses_fallbacks(monkeypatch):
    calls = []

    def fake_web_search(query, max_results=10, ctx=None):
        calls.append(query)
        if len(calls) <= 2:
            return {
                "results": [
                    {"title": "A", "url": "https://a", "snippet": "one"},
                    {"title": "A duplicate", "url": "https://a", "snippet": "dup"},
                    {"title": "B", "url": "", "snippet": "two"},
                ]
            }
        return {"results": []}

    monkeypatch.setattr("biobank_agent.skills.web_search.web_search", fake_web_search)
    monkeypatch.setattr(deep_mod, "_search_europe_pmc", lambda topic, max_results: [{"title": "PMC", "url": "https://pmc", "snippet": "pmc"}])
    ctx = SimpleNamespace(settings=SimpleNamespace(biobank_name="UK Biobank"))

    results = deep_mod._search_literature("HbA1c diabetes", 5, ctx)

    assert calls[:2] == ["HbA1c diabetes scientific study", "HbA1c diabetes UK Biobank"]
    assert [r["url"] or r["title"] for r in results] == ["https://a", "B"]

    monkeypatch.setattr("biobank_agent.skills.web_search.web_search", lambda *args, **kwargs: {"results": []})
    fallback = deep_mod._search_literature("no hits", 3, ctx)
    assert fallback[0] == {"title": "PMC", "url": "https://pmc", "snippet": "pmc", "doi": ""}

    monkeypatch.setattr("biobank_agent.skills.web_search.web_search", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("search broken")))
    assert deep_mod._search_literature("topic", 3, ctx) == []


def test_search_literature_filters_ukb_noise_and_preserves_doi(monkeypatch):
    def fake_web_search(query, max_results=10, ctx=None):
        return {
            "results": [
                {
                    "title": "变压器的短路电压百分比值",
                    "url": "https://zhidao.baidu.com/question/1.html",
                    "snippet": "Uk% transformer text",
                },
                {
                    "title": "United Kingdom statistics",
                    "url": "https://www.statista.com/topics/755/uk/",
                    "snippet": "UK facts",
                },
                {
                    "title": "Disease prediction with multi-omics and biomarkers empowers case-control genetic discoveries in the UK Biobank",
                    "url": "https://www.nature.com/articles/s41588-024-01898-1",
                    "snippet": "DOI 10.1038/s41588-024-01898-1",
                },
            ]
        }

    monkeypatch.setattr("biobank_agent.skills.web_search.web_search", fake_web_search)
    monkeypatch.setattr(deep_mod, "_search_europe_pmc", lambda topic, max_results: [])
    ctx = SimpleNamespace(settings=SimpleNamespace(biobank_name="UK Biobank"))

    results = deep_mod._search_literature(
        "UK Biobank plasma proteomics Olink risk prediction key papers",
        5,
        ctx,
    )

    urls = [r["url"] for r in results]
    assert "https://zhidao.baidu.com/question/1.html" not in urls
    assert "https://www.statista.com/topics/755/uk/" not in urls
    assert results[0]["doi"] == "10.1038/s41588-024-01898-1"
    assert any(r["doi"] == "10.1038/s41586-023-06592-6" for r in results)
    assert deep_mod._needs_biobank_literature_filter("plain topic") is False
    assert deep_mod._curated_sources_for_topic("UK Biobank demographics", 3) == []


def test_search_literature_expands_sparse_ukb_search(monkeypatch):
    calls = []

    def fake_web_search(query, max_results=10, ctx=None):
        calls.append(query)
        return {"results": []}

    monkeypatch.setattr("biobank_agent.skills.web_search.web_search", fake_web_search)
    monkeypatch.setattr(
        deep_mod,
        "_search_europe_pmc",
        lambda topic, max_results: [
            {"title": f"PMC {i}", "url": f"https://europepmc.org/{i}", "snippet": "UK Biobank biomarker cohort"}
            for i in range(6)
        ],
    )
    ctx = SimpleNamespace(settings=SimpleNamespace(biobank_name="UK Biobank"))

    results = deep_mod._search_literature(
        "UK Biobank longitudinal biomarker trajectories HealthFormer disease progression forecast",
        10,
        ctx,
    )

    assert len(calls) >= 5
    assert any("PubMed" in q for q in calls)
    assert any(r["title"].startswith("PMC") for r in results)


def test_search_literature_breaks_after_fallback_and_skips_empty_keys(monkeypatch):
    calls = []

    def fake_web_search(query, max_results=10, ctx=None):
        calls.append(query)
        if len(calls) <= 2:
            return {"results": []}
        return {
            "results": [
                {"title": "", "url": "", "snippet": "skip"},
                {"title": "Fallback hit", "url": "https://fallback", "snippet": "hit"},
            ]
        }

    monkeypatch.setattr("biobank_agent.skills.web_search.web_search", fake_web_search)
    monkeypatch.setattr(deep_mod, "_search_europe_pmc", lambda topic, max_results: [])

    results = deep_mod._search_literature("topic", 1, SimpleNamespace())

    assert len(calls) == 3
    assert results == [{"title": "Fallback hit", "url": "https://fallback", "snippet": "hit", "doi": ""}]


def test_fetch_abstracts_and_cross_reference(monkeypatch, tmp_path):
    sources = [
        {"title": "Paper A", "url": "https://a", "snippet": "old"},
        {"title": "Paper B", "url": "https://b", "snippet": "old"},
        {"title": "Paper C", "url": "https://c", "snippet": "old"},
    ]

    def fake_fetch(url, max_chars=2000, ctx=None):
        if url.endswith("b"):
            raise RuntimeError("offline")
        if url.endswith("c"):
            return {"error": "blocked"}
        return {"title": "Fetched A", "content": "Fetched abstract"}

    monkeypatch.setattr("biobank_agent.skills.web_fetch.web_fetch", fake_fetch)
    enriched = deep_mod._fetch_abstracts(sources, max_fetch=3, ctx=None)
    no_fetch = deep_mod._fetch_abstracts([{"title": "No URL"}], max_fetch=0, ctx=None)

    assert enriched[0]["title"] == "Fetched A"
    assert enriched[0]["abstract"] == "Fetched abstract"
    assert "abstract" not in enriched[1]
    assert no_fetch == [{"title": "No URL"}]

    real_import = builtins.__import__

    def no_web_fetch(name, *args, **kwargs):
        if name == "biobank_agent.skills.web_fetch":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_web_fetch)
    assert deep_mod._fetch_abstracts(sources, max_fetch=1, ctx=None) is sources

    class FakeCatalog:
        def search(self, keyword, limit=10):
            if keyword == "broken":
                raise RuntimeError("catalog down")
            return [
                {"field_id": "30740", "title": "Glucose", "category_id": "bio", "value_type": "continuous"},
                {"field_id": "30740", "title": "Duplicate", "category_id": "bio", "value_type": "continuous"},
                {"field_id": "30750", "title": "HbA1c", "category_id": "bad", "value_type": "continuous"},
            ]

        def category_name(self, category_id):
            if category_id == "bad":
                raise RuntimeError("bad category")
            return "Biochemistry"

    ctx = SimpleNamespace(catalog=FakeCatalog())
    fields = deep_mod._cross_reference_biobank_fields("glucose broken cardiometabolic", ctx)

    assert fields == [
        {"field_id": "30740", "title": "Glucose", "category": "Biochemistry", "value_type": "continuous"},
        {"field_id": "30750", "title": "HbA1c", "category": "", "value_type": "continuous"},
    ]
    assert deep_mod._cross_reference_biobank_fields("topic", ctx=None) == []
    assert deep_mod._cross_reference_biobank_fields(None, ctx) == []


def test_compile_brief_and_deep_research_save_paths(tmp_path, monkeypatch):
    sources = [
        {"title": "A paper", "url": "https://doi.org/10.1038/example", "doi": "10.1038/example", "snippet": "x" * 350},
    ]
    fields = [{"field_id": "30740", "title": "Glucose", "category": "Biochemistry", "value_type": "continuous"}]

    brief = deep_mod._compile_brief("diabetes", sources, fields, bank_name="UK Biobank")
    empty_brief = deep_mod._compile_brief("rare topic", [], [], bank_name="UK Biobank")

    assert "Sources reviewed**: 1" in brief
    assert "**DOI**: 10.1038/example" in brief
    assert "xxx..." in brief
    assert "| 30740 | Glucose | Biochemistry | continuous |" in brief
    assert "No sources found" in empty_brief
    assert "Manual catalogue review recommended" in empty_brief

    short_no_url = deep_mod._compile_brief(
        "short source",
        [{"title": "No URL", "snippet": "short snippet"}],
        [],
        bank_name="UK Biobank",
    )
    assert "short snippet" in short_no_url
    assert "**URL**" not in short_no_url

    ctx = SimpleNamespace(
        report_dir=tmp_path,
        settings=SimpleNamespace(biobank_name="UK Biobank"),
    )
    monkeypatch.setattr(deep_mod, "_search_literature", lambda topic, max_sources, ctx: sources)
    monkeypatch.setattr(deep_mod, "_fetch_abstracts", lambda sources, max_fetch, ctx: sources)
    monkeypatch.setattr(deep_mod, "_cross_reference_biobank_fields", lambda topic, ctx: fields)

    result = deep_mod.deep_research("diabetes/biomarkers?", max_sources=99, ctx=ctx)

    assert result["topic"] == "diabetes/biomarkers?"
    assert result["n_sources"] == 1
    assert result["status"] == "PARTIAL"
    assert result["min_expected_sources"] == 3
    assert result["search_attempts"]
    assert result["sources"][0]["doi"] == "10.1038/example"
    assert result["n_biobank_fields"] == 1
    assert result["brief_path"] is not None
    assert "diabetesbiomarkers" in result["brief_path"]
    assert (tmp_path / "research").exists()

    def bad_write_text(self, text, encoding=None):
        raise RuntimeError("read-only")

    monkeypatch.setattr(deep_mod.Path, "write_text", bad_write_text)
    unsaved = deep_mod.deep_research("topic", max_sources=0, ctx=ctx)
    assert unsaved["brief_path"] is None


def test_deep_research_no_sources_uses_default_research_dir(monkeypatch):
    monkeypatch.setattr(deep_mod, "_search_literature", lambda topic, max_sources, ctx: [])
    monkeypatch.setattr(deep_mod, "_cross_reference_biobank_fields", lambda topic, ctx: [])
    monkeypatch.setattr(deep_mod.Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
    monkeypatch.setattr(deep_mod.Path, "write_text", lambda self, text, encoding=None: None)

    result = deep_mod.deep_research("empty topic", max_sources=3, ctx=None)

    assert result["n_sources"] == 0
    assert result["n_biobank_fields"] == 0
    assert result["brief_path"] is not None
    assert "research_empty_topic" in result["brief_path"]
