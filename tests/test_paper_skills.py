"""Tests for paper fetching and structured paper-reading skills."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from biobank_agent.skills import fetch_paper as fetch_mod
from biobank_agent.skills import read_paper as read_mod


class FakeCtx:
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.settings = SimpleNamespace(
            biobank_name="UK Biobank",
            biobank_abbreviation="UKB",
        )


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", headers=None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._json_data


def test_identifier_detection_and_clean_abstract():
    assert fetch_mod._detect_identifier_type("10.1038/s41588-024-01898-1") == "doi"
    assert fetch_mod._detect_identifier_type("https://doi.org/10.1038/test") == "url"
    assert fetch_mod._detect_identifier_type("doi.org/10.1038/test") == "url"
    assert fetch_mod._detect_identifier_type("https://example.org/paper.pdf") == "url"
    assert fetch_mod._detect_identifier_type("A cohort study of biomarkers") == "title"

    assert fetch_mod._clean_abstract("<jats:p>HbA1c predicts diabetes.</jats:p>") == "HbA1c predicts diabetes."
    assert fetch_mod._clean_abstract("") == ""


def test_resolve_doi_maps_csl_metadata(monkeypatch):
    def fake_get(url, **kwargs):
        assert kwargs["headers"]["Accept"] == "application/vnd.citationstyles.csl+json"
        return FakeResponse(
            json_data={
                "title": "Biomarker Study",
                "author": [{"given": "Ada", "family": "Lovelace"}, {"family": "Curie"}],
                "abstract": "<p>Structured abstract.</p>",
                "URL": "https://publisher.example/article",
                "container-title": "Nature Medicine",
                "issued": {"date-parts": [[2025, 1, 1]]},
            }
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=fake_get))

    result = fetch_mod._resolve_doi("10.1038/example")

    assert result["doi"] == "10.1038/example"
    assert result["title"] == "Biomarker Study"
    assert result["authors"] == ["Ada Lovelace", "Curie"]
    assert result["abstract"] == "Structured abstract."
    assert result["journal"] == "Nature Medicine"
    assert result["year"] == 2025


def test_resolve_doi_skips_empty_issued_date(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(
            get=lambda *args, **kwargs: FakeResponse(
                json_data={
                    "title": "No Year",
                    "issued": {"date-parts": [[]]},
                }
            )
        ),
    )

    result = fetch_mod._resolve_doi("10.1038/no-year")

    assert result["title"] == "No Year"
    assert "year" not in result


def test_resolve_doi_non_200_and_exception_paths(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(get=lambda *args, **kwargs: FakeResponse(status_code=503)),
    )
    unavailable = fetch_mod._resolve_doi("10.1038/unavailable")
    assert unavailable == {"doi": "10.1038/unavailable"}

    def raising_get(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=raising_get))
    failed = fetch_mod._resolve_doi("10.1038/fail")
    assert failed["url"] == "https://doi.org/10.1038/fail"


def test_try_download_pdf_saves_sanitized_filename(tmp_path, monkeypatch):
    def fake_get(url, **kwargs):
        return FakeResponse(
            content=b"%PDF test",
            headers={"content-type": "application/pdf; charset=binary"},
        )

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=fake_get))

    path = fetch_mod._try_download_pdf("https://example.org/files/report name?x=1", tmp_path)

    assert path is not None
    assert path.name == "report_name_x_1.pdf"
    assert path.read_bytes() == b"%PDF test"


def test_try_download_pdf_keeps_existing_pdf_suffix(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(
            get=lambda *args, **kwargs: FakeResponse(
                content=b"%PDF test",
                headers={"content-type": "application/pdf"},
            )
        ),
    )

    path = fetch_mod._try_download_pdf("https://example.org/paper.pdf", tmp_path)

    assert path is not None
    assert path.name == "paper.pdf"


def test_try_download_pdf_returns_none_for_non_pdf_and_exceptions(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(get=lambda *args, **kwargs: FakeResponse(headers={"content-type": "text/html"})),
    )
    assert fetch_mod._try_download_pdf("https://example.org/article", tmp_path) is None

    def raising_get(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=raising_get))
    assert fetch_mod._try_download_pdf("https://example.org/article.pdf", tmp_path) is None


def test_fetch_pdf_from_doi_prefers_unpaywall_pdf(monkeypatch, tmp_path):
    tried = []

    def fake_get(url, **kwargs):
        assert url == "https://api.unpaywall.org/v2/10.1038/example"
        return FakeResponse(json_data={"best_oa_location": {"url_for_pdf": "https://oa.example/paper.pdf"}})

    def fake_download(url, save_dir):
        tried.append(url)
        if url == "https://oa.example/paper.pdf":
            return save_dir / "paper.pdf"
        return None

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=fake_get))
    monkeypatch.setattr(fetch_mod, "_try_download_pdf", fake_download)

    path = fetch_mod._fetch_pdf_from_doi("10.1038/example", tmp_path)

    assert path == tmp_path / "paper.pdf"
    assert tried[0] == "https://oa.example/paper.pdf"


def test_fetch_pdf_from_doi_uses_oa_url_fallback_and_none_paths(monkeypatch, tmp_path):
    tried = []

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(
            get=lambda *args, **kwargs: FakeResponse(
                json_data={"best_oa_location": {"url": "https://oa.example/article"}}
            )
        ),
    )
    monkeypatch.setattr(fetch_mod, "_try_download_pdf", lambda url, save_dir: tried.append(url) or None)

    assert fetch_mod._fetch_pdf_from_doi("10.1038/no-pdf", tmp_path) is None
    assert tried[0] == "https://oa.example/article"
    assert tried[-1].startswith("https://europepmc.org/")

    tried.clear()

    def raising_get(*args, **kwargs):
        raise RuntimeError("unpaywall offline")

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=raising_get))
    assert fetch_mod._fetch_pdf_from_doi("10.1038/offline", tmp_path) is None
    assert tried[0] == "https://doi.org/10.1038/offline"


def test_fetch_pdf_from_doi_unpaywall_non_200_and_no_pdf_url(monkeypatch, tmp_path):
    calls = []

    def fake_get_non_200(url, **kwargs):
        calls.append(url)
        if "unpaywall" in url:
            return FakeResponse(status_code=503)
        return FakeResponse(status_code=404)

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=fake_get_non_200))
    assert fetch_mod._fetch_pdf_from_doi("10.1038/non-oa", tmp_path) is None
    assert any("unpaywall" in url for url in calls)

    def fake_get_no_pdf(url, **kwargs):
        if "unpaywall" in url:
            return FakeResponse(json_data={"best_oa_location": {}})
        return FakeResponse(status_code=404)

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=fake_get_no_pdf))
    assert fetch_mod._fetch_pdf_from_doi("10.1038/no-pdf-url", tmp_path) is None


def test_fetch_paper_doi_extracts_pdf_text_and_metadata(tmp_path, monkeypatch):
    ctx = FakeCtx(tmp_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF")

    monkeypatch.setattr(
        fetch_mod,
        "_resolve_doi",
        lambda doi: {"title": "Publisher Title", "authors": ["A Author"], "abstract": "Abstract", "doi": doi},
    )
    monkeypatch.setattr(fetch_mod, "_fetch_pdf_from_doi", lambda doi, save_dir: pdf_path)

    def fake_read_pdf(path, max_pages=0, ctx=None):
        assert path == str(pdf_path)
        return {
            "text": "Full paper text with E11 and UKB field 30750.",
            "n_pages": 3,
            "metadata": {"title": "PDF Title", "author": "PDF Author"},
        }

    monkeypatch.setattr("biobank_agent.skills.read_pdf.read_pdf", fake_read_pdf)

    result = fetch_mod.fetch_paper("10.1038/example", ctx=ctx)

    assert result["source"] == "doi"
    assert result["title"] == "Publisher Title"
    assert result["full_text"].startswith("Full paper text")
    assert result["n_pages"] == 3
    assert result["pdf_path"] == str(pdf_path)


def test_fetch_paper_uses_default_papers_dir_without_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch_mod, "_resolve_doi", lambda doi: {"doi": doi})
    monkeypatch.setattr(fetch_mod, "_fetch_pdf_from_doi", lambda doi, save_dir: None)

    result = fetch_mod.fetch_paper("10.1038/default")

    assert result["warning"]
    assert (tmp_path / "papers").is_dir()


def test_fetch_paper_url_uses_direct_pdf_and_web_fallback(tmp_path, monkeypatch):
    ctx = FakeCtx(tmp_path)
    downloaded = tmp_path / "direct.pdf"
    monkeypatch.setattr(fetch_mod, "_try_download_pdf", lambda url, save_dir: downloaded)
    monkeypatch.setattr(
        "biobank_agent.skills.read_pdf.read_pdf",
        lambda path, max_pages=0, ctx=None: {"text": "", "metadata": {}},
    )

    direct = fetch_mod.fetch_paper("https://example.org/direct.pdf", ctx=ctx)

    assert direct["source"] == "url"
    assert direct["pdf_path"] == str(downloaded)

    monkeypatch.setattr(fetch_mod, "_try_download_pdf", lambda url, save_dir: None)
    monkeypatch.setattr(
        "biobank_agent.skills.web_fetch.web_fetch",
        lambda url, max_chars=8000, ctx=None: {"title": "Fetched page", "content": "Long abstract " * 300},
    )

    fallback = fetch_mod.fetch_paper("https://example.org/article", ctx=ctx)

    assert fallback["title"] == "Fetched page"
    assert len(fallback["abstract"]) == 2000


def test_fetch_paper_url_doi_no_pdf_direct_pdf_and_empty_web_fetch(tmp_path, monkeypatch):
    ctx = FakeCtx(tmp_path)
    direct_pdf = tmp_path / "direct.pdf"
    monkeypatch.setattr(fetch_mod, "_resolve_doi", lambda doi: {"title": "URL DOI", "doi": doi})
    monkeypatch.setattr(fetch_mod, "_fetch_pdf_from_doi", lambda doi, save_dir: None)
    monkeypatch.setattr(fetch_mod, "_try_download_pdf", lambda url, save_dir: direct_pdf)
    monkeypatch.setattr(
        "biobank_agent.skills.read_pdf.read_pdf",
        lambda path, max_pages=0, ctx=None: {"text": "", "metadata": {}},
    )

    direct = fetch_mod.fetch_paper("https://example.org/10.1038/url-direct.pdf", ctx=ctx)

    assert direct["doi"] == "10.1038/url-direct.pdf"
    assert direct["pdf_path"] == str(direct_pdf)

    monkeypatch.setattr(fetch_mod, "_try_download_pdf", lambda url, save_dir: None)
    monkeypatch.setattr("biobank_agent.skills.web_fetch.web_fetch", lambda **kwargs: {"title": "No content"})
    empty_fetch = fetch_mod.fetch_paper("https://example.org/no-content", ctx=ctx)

    assert empty_fetch["abstract"] == ""
    assert "paywall" in empty_fetch["warning"]


def test_fetch_paper_direct_pdf_download_failure_uses_web_fallback(tmp_path, monkeypatch):
    ctx = FakeCtx(tmp_path)
    monkeypatch.setattr(fetch_mod, "_try_download_pdf", lambda url, save_dir: None)
    monkeypatch.setattr(
        "biobank_agent.skills.web_fetch.web_fetch",
        lambda url, max_chars=8000, ctx=None: {"title": "PDF Landing", "content": "landing page"},
    )

    result = fetch_mod.fetch_paper("https://example.org/missing.pdf", ctx=ctx)

    assert result["pdf_path"] is None
    assert result["title"] == "PDF Landing"
    assert result["abstract"] == "landing page"


def test_fetch_paper_url_doi_and_web_fallback_error_paths(tmp_path, monkeypatch):
    ctx = FakeCtx(tmp_path)
    pdf_path = tmp_path / "doi.pdf"
    monkeypatch.setattr(fetch_mod, "_resolve_doi", lambda doi: {"title": "URL DOI", "doi": doi})
    monkeypatch.setattr(fetch_mod, "_fetch_pdf_from_doi", lambda doi, save_dir: pdf_path)
    monkeypatch.setattr(
        "biobank_agent.skills.read_pdf.read_pdf",
        lambda path, max_pages=0, ctx=None: {"text": "URL DOI full text", "metadata": {}},
    )

    doi_url = fetch_mod.fetch_paper("https://doi.org/10.1038/url-doi?x=1", ctx=ctx)

    assert doi_url["doi"] == "10.1038/url-doi"
    assert doi_url["full_text"] == "URL DOI full text"

    monkeypatch.setattr(fetch_mod, "_fetch_pdf_from_doi", lambda doi, save_dir: None)
    monkeypatch.setattr(fetch_mod, "_try_download_pdf", lambda url, save_dir: None)

    def broken_fetch(*args, **kwargs):
        raise RuntimeError("fetch offline")

    monkeypatch.setattr("biobank_agent.skills.web_fetch.web_fetch", broken_fetch)
    failed = fetch_mod.fetch_paper("https://example.org/no-fetch", ctx=ctx)
    assert "paywall" in failed["warning"]


def test_fetch_paper_title_search_uses_first_doi_hit(tmp_path, monkeypatch):
    ctx = FakeCtx(tmp_path)
    pdf_path = tmp_path / "oa.pdf"
    monkeypatch.setattr(
        "biobank_agent.skills.web_search.web_search",
        lambda query, max_results=5, ctx=None: {
            "results": [
                {
                    "title": "Search title",
                    "url": "https://example.org/article",
                    "snippet": "doi 10.1038/search",
                }
            ]
        },
    )
    monkeypatch.setattr(fetch_mod, "_resolve_doi", lambda doi: {"title": "Resolved title", "doi": doi})
    monkeypatch.setattr(fetch_mod, "_fetch_pdf_from_doi", lambda doi, save_dir: pdf_path)
    monkeypatch.setattr("biobank_agent.skills.read_pdf.read_pdf", lambda **kwargs: {"text": "Paper text"})

    result = fetch_mod.fetch_paper("searchable biomarker paper", ctx=ctx)

    assert result["source"] == "title"
    assert result["title"] == "Resolved title"
    assert result["doi"] == "10.1038/search"
    assert result["full_text"] == "Paper text"


def test_fetch_paper_title_search_no_hits_and_exception(tmp_path, monkeypatch):
    ctx = FakeCtx(tmp_path)
    monkeypatch.setattr(
        "biobank_agent.skills.web_search.web_search",
        lambda query, max_results=5, ctx=None: {"results": []},
    )

    no_hits = fetch_mod.fetch_paper("unknown paper", ctx=ctx)
    assert no_hits["source"] == "title"
    assert "paywall" in no_hits["warning"]

    def broken_search(*args, **kwargs):
        raise RuntimeError("search offline")

    monkeypatch.setattr("biobank_agent.skills.web_search.web_search", broken_search)
    failed = fetch_mod.fetch_paper("broken paper", ctx=ctx)
    assert "Could not find paper by title" in failed["error"]


def test_fetch_paper_handles_unexpected_identifier_type(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_mod, "_detect_identifier_type", lambda identifier: "other")

    result = fetch_mod.fetch_paper("opaque", ctx=FakeCtx(tmp_path))

    assert result["source"] == "other"
    assert "warning" in result


def test_fetch_paper_title_search_hits_without_doi_and_doi_without_pdf(tmp_path, monkeypatch):
    ctx = FakeCtx(tmp_path)
    monkeypatch.setattr(
        "biobank_agent.skills.web_search.web_search",
        lambda query, max_results=5, ctx=None: {
            "results": [
                {"title": "No DOI", "url": "https://example.org/a", "snippet": "plain"},
                {"title": "Still no DOI", "url": "https://example.org/b", "snippet": "plain"},
            ]
        },
    )

    no_doi = fetch_mod.fetch_paper("plain title", ctx=ctx)
    assert no_doi["title"] == "No DOI"
    assert no_doi["doi"] == ""

    monkeypatch.setattr(
        "biobank_agent.skills.web_search.web_search",
        lambda query, max_results=5, ctx=None: {
            "results": [
                {"title": "First no DOI", "url": "https://example.org/a", "snippet": "plain"},
                {"title": "DOI hit", "url": "https://example.org/10.1038/no-pdf", "snippet": "doi"},
            ]
        },
    )
    monkeypatch.setattr(fetch_mod, "_resolve_doi", lambda doi: {"doi": doi, "title": "Resolved No PDF"})
    monkeypatch.setattr(fetch_mod, "_fetch_pdf_from_doi", lambda doi, save_dir: None)

    doi_no_pdf = fetch_mod.fetch_paper("doi no pdf title", ctx=ctx)
    assert doi_no_pdf["doi"] == "10.1038/no-pdf"
    assert doi_no_pdf["pdf_path"] is None


def test_fetch_paper_pdf_metadata_fallback_and_read_error(tmp_path, monkeypatch):
    ctx = FakeCtx(tmp_path)
    pdf_path = tmp_path / "metadata.pdf"
    monkeypatch.setattr(fetch_mod, "_resolve_doi", lambda doi: {"doi": doi})
    monkeypatch.setattr(fetch_mod, "_fetch_pdf_from_doi", lambda doi, save_dir: pdf_path)
    monkeypatch.setattr(
        "biobank_agent.skills.read_pdf.read_pdf",
        lambda path, max_pages=0, ctx=None: {
            "text": "Extracted text",
            "metadata": {"title": "PDF title", "author": "PDF author"},
        },
    )

    result = fetch_mod.fetch_paper("10.1038/pdf-meta", ctx=ctx)

    assert result["title"] == "PDF title"
    assert result["authors"] == ["PDF author"]

    def broken_read_pdf(*args, **kwargs):
        raise RuntimeError("bad pdf")

    monkeypatch.setattr("biobank_agent.skills.read_pdf.read_pdf", broken_read_pdf)
    failed_read = fetch_mod.fetch_paper("10.1038/pdf-read-error", ctx=ctx)
    assert failed_read["pdf_path"] == str(pdf_path)
    assert "paywall" in failed_read["warning"]


def test_fetch_paper_warning_when_no_text_or_abstract(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_mod, "_resolve_doi", lambda doi: {"doi": doi})
    monkeypatch.setattr(fetch_mod, "_fetch_pdf_from_doi", lambda doi, save_dir: None)

    result = fetch_mod.fetch_paper("10.1038/paywalled", ctx=FakeCtx(tmp_path))

    assert "paywall" in result["warning"]


def test_read_paper_extractors_and_context_aware_template(tmp_path):
    ctx = FakeCtx(tmp_path)
    same_name_ctx = SimpleNamespace(
        settings=SimpleNamespace(biobank_name="UKB", biobank_abbreviation="UKB")
    )
    text = "Cases included E11, I10 and E11.9. We used UKB field ID 30750 and UK Biobank 21001."

    assert read_mod._get_field_re().search("UK Biobank field ID 21001").group(1) == "21001"
    assert read_mod._get_field_re(same_name_ctx).search("UKB field ID 30750").group(1) == "30750"
    assert read_mod._extract_icd10_codes("E11 U12 W19") == ["E11", "W19"]
    assert read_mod._extract_icd10_codes(text) == ["E11", "I10", "E11.9"]
    assert read_mod._extract_biobank_field_refs(text, ctx=ctx) == ["30750", "21001"]

    template = read_mod._build_analysis_template("ukb-relevance", ctx=ctx)
    assert "UK Biobank Relevance" in template
    assert "Uses UKB Data" in template


def test_read_paper_doi_path_truncates_and_extracts_fields(tmp_path, monkeypatch):
    ctx = FakeCtx(tmp_path)
    long_text = ("E11 and UKB field 30750. " * 500)
    monkeypatch.setattr(
        "biobank_agent.skills.fetch_paper.fetch_paper",
        lambda identifier, ctx=None: {
            "title": "Fetched title",
            "authors": ["A Author"],
            "doi": "10.1038/example",
            "full_text": long_text,
            "pdf_path": "/tmp/paper.pdf",
        },
    )

    result = read_mod.read_paper("10.1038/example", focus="methods", ctx=ctx)

    assert result["summary"]["title"] == "Fetched title"
    assert result["text_truncated"] is True
    assert result["full_text_length"] == len(long_text)
    assert result["paper_text"].endswith("[... paper text truncated for analysis ...]")
    assert result["icd10_codes_mentioned"] == ["E11"]
    assert result["biobank_fields_mentioned"] == ["30750"]
    assert "methodology" in result["analysis_template"]


def test_read_paper_file_path_and_error_paths(tmp_path, monkeypatch):
    ctx = FakeCtx(tmp_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF")

    monkeypatch.setattr(
        "biobank_agent.skills.read_pdf.read_pdf",
        lambda path, max_pages=0, ctx=None: {
            "text": "Paper mentions I25 and UKB 21001.",
            "n_pages": 4,
            "metadata": {"title": "PDF title", "author": "Analyst"},
        },
    )

    result = read_mod.read_paper(str(pdf_path), ctx=ctx)

    assert result["summary"]["title"] == "PDF title"
    assert result["summary"]["authors"] == ["Analyst"]
    assert result["icd10_codes_mentioned"] == ["I25"]
    assert result["biobank_fields_mentioned"] == ["21001"]
    assert result["focus"] == "general"

    monkeypatch.setattr("biobank_agent.skills.read_pdf.read_pdf", lambda **kwargs: {"text": ""})
    no_text = read_mod.read_paper(str(pdf_path), ctx=ctx)
    assert no_text["error"] == "No text could be extracted from the paper."

    monkeypatch.setattr("biobank_agent.skills.fetch_paper.fetch_paper", lambda **kwargs: {"warning": "closed"})
    failed = read_mod.read_paper("https://example.org/closed", ctx=ctx)
    assert failed["summary"]["access_warning"] == "closed"
    assert failed["error"] == "No text could be extracted from the paper."


def test_read_paper_fetch_and_pdf_failure_paths(tmp_path, monkeypatch):
    ctx = FakeCtx(tmp_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF")

    monkeypatch.setattr(
        "biobank_agent.skills.fetch_paper.fetch_paper",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    assert read_mod.read_paper("10.1038/fail", ctx=ctx) == {"error": "Failed to fetch paper: network down"}

    monkeypatch.setattr(
        "biobank_agent.skills.read_pdf.read_pdf",
        lambda **kwargs: {"error": "encrypted PDF"},
    )
    assert read_mod.read_paper(str(pdf_path), ctx=ctx) == {"error": "encrypted PDF"}

    monkeypatch.setattr(
        "biobank_agent.skills.read_pdf.read_pdf",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad xref")),
    )
    assert read_mod.read_paper(str(pdf_path), ctx=ctx) == {"error": "Failed to read PDF: bad xref"}
