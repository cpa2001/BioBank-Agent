"""Fetch paper skill — download and extract scientific papers by DOI, URL, or title.

Resolves identifiers, downloads PDFs when accessible, and extracts structured
content including title, authors, abstract, and full text.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)

# Regex for DOI detection (e.g. 10.1038/s41588-024-01898-1)
_DOI_RE = re.compile(r"^10\.\d{4,9}/[^\s]+$")


# ---------------------------------------------------------------------------
# Identifier detection
# ---------------------------------------------------------------------------

def _detect_identifier_type(identifier: str) -> str:
    """Return ``'doi'``, ``'url'``, or ``'title'``."""
    identifier = identifier.strip()
    if _DOI_RE.match(identifier):
        return "doi"
    if identifier.startswith("http://") or identifier.startswith("https://"):
        return "url"
    # Could also be a DOI URL like https://doi.org/10.xxxx
    if "doi.org/" in identifier:
        return "url"
    return "title"


# ---------------------------------------------------------------------------
# DOI resolution and PDF fetching
# ---------------------------------------------------------------------------

def _resolve_doi(doi: str) -> dict:
    """Resolve a DOI via doi.org content negotiation for metadata + landing page."""
    import httpx

    metadata: dict[str, Any] = {"doi": doi}

    # Try content negotiation for JSON metadata
    try:
        resp = httpx.get(
            f"https://doi.org/{doi}",
            headers={"Accept": "application/vnd.citationstyles.csl+json"},
            follow_redirects=True,
            timeout=20,
        )
        if resp.status_code == 200:
            data = resp.json()
            metadata["title"] = data.get("title", "")
            authors = data.get("author", [])
            metadata["authors"] = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in authors
            ]
            metadata["abstract"] = _clean_abstract(data.get("abstract", ""))
            metadata["url"] = data.get("URL", f"https://doi.org/{doi}")
            metadata["journal"] = data.get("container-title", "")
            issued = data.get("issued", {})
            date_parts = issued.get("date-parts", [[]])
            if date_parts and date_parts[0]:
                metadata["year"] = date_parts[0][0]
    except Exception as exc:
        logger.warning("DOI content negotiation failed for %s: %s", doi, exc)
        metadata["url"] = f"https://doi.org/{doi}"

    return metadata


def _clean_abstract(raw: str) -> str:
    """Strip HTML/JATS tags from an abstract string."""
    if not raw:
        return ""
    cleaned = re.sub(r"<[^>]+>", "", raw)
    return cleaned.strip()


def _try_download_pdf(url: str, save_dir: Path) -> Path | None:
    """Attempt to download a PDF from *url* and save to *save_dir*.

    Returns the saved path, or None if not accessible.
    """
    import httpx

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BiobankAgent/0.1)",
        "Accept": "application/pdf,*/*",
    }
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=60, headers=headers)
        content_type = resp.headers.get("content-type", "")
        if resp.status_code == 200 and "pdf" in content_type.lower():
            # Derive filename from URL
            fname = url.rstrip("/").split("/")[-1]
            if not fname.endswith(".pdf"):
                fname += ".pdf"
            # Sanitise filename
            fname = re.sub(r'[^\w.\-]', '_', fname)
            save_path = save_dir / fname
            save_path.write_bytes(resp.content)
            return save_path
    except Exception as exc:
        logger.debug("PDF download failed from %s: %s", url, exc)
    return None


def _fetch_pdf_from_doi(doi: str, save_dir: Path) -> Path | None:
    """Try common open-access PDF sources for a given DOI."""
    candidates = [
        # Sci-Hub alternatives / open-access mirrors
        f"https://doi.org/{doi}",
        # PubMed Central (via DOI redirect)
        f"https://europepmc.org/backend/ptpmcrender.fcgi?accid=doi:{doi}&blobtype=pdf",
    ]

    # Try Unpaywall for legal open-access URL
    try:
        import httpx

        resp = httpx.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": "biobank-agent@example.com"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            oa_loc = data.get("best_oa_location") or {}
            pdf_url = oa_loc.get("url_for_pdf") or oa_loc.get("url")
            if pdf_url:
                candidates.insert(0, pdf_url)
    except Exception:
        pass

    for url in candidates:
        pdf_path = _try_download_pdf(url, save_dir)
        if pdf_path is not None:
            return pdf_path

    return None


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

@skill(
    name="fetch_paper",
    description="Download and process a scientific paper by DOI, URL, or title. "
                "Resolves the identifier, downloads PDF, and extracts structured content.",
    parameters={
        "identifier": {
            "type": "string",
            "description": "DOI (e.g., '10.1038/s41588-024-01898-1'), URL, or paper title",
        },
    },
    required=["identifier"],
)
def fetch_paper(identifier: str, *, ctx=None) -> dict:
    """Fetch a scientific paper and extract its content.

    The function tries, in order:

    1. Resolve metadata (title, authors, abstract) via DOI content negotiation.
    2. Download the PDF from open-access sources (Unpaywall, publisher).
    3. Extract full text from the PDF using PyMuPDF.

    If the PDF is paywalled, the abstract is returned instead.

    Returns
    -------
    dict
        ``{"title": ..., "authors": [...], "abstract": ..., "full_text": ...,
           "pdf_path": ..., "doi": ...}``
    """
    identifier = identifier.strip()
    id_type = _detect_identifier_type(identifier)

    # Determine save directory
    if ctx is not None and hasattr(ctx, "report_dir"):
        papers_dir = Path(ctx.report_dir) / "papers"
    else:
        papers_dir = Path("./papers")
    papers_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "title": "",
        "authors": [],
        "abstract": "",
        "full_text": "",
        "pdf_path": None,
        "doi": "",
        "source": id_type,
    }

    # ------------------------------------------------------------------
    # Route by identifier type
    # ------------------------------------------------------------------

    if id_type == "doi":
        doi = identifier
        result["doi"] = doi

        # Metadata via content negotiation
        meta = _resolve_doi(doi)
        result.update({k: v for k, v in meta.items() if v})

        # Try to download PDF
        pdf_path = _fetch_pdf_from_doi(doi, papers_dir)
        if pdf_path:
            result["pdf_path"] = str(pdf_path)

    elif id_type == "url":
        result["url"] = identifier

        # Check if URL contains a DOI we can extract
        doi_match = re.search(r"(10\.\d{4,9}/[^\s&?#]+)", identifier)
        if doi_match:
            doi = doi_match.group(1)
            result["doi"] = doi
            meta = _resolve_doi(doi)
            result.update({k: v for k, v in meta.items() if v})
            pdf_path = _fetch_pdf_from_doi(doi, papers_dir)
            if pdf_path:
                result["pdf_path"] = str(pdf_path)

        # If URL ends with .pdf, try direct download
        if result["pdf_path"] is None and identifier.lower().endswith(".pdf"):
            pdf_path = _try_download_pdf(identifier, papers_dir)
            if pdf_path:
                result["pdf_path"] = str(pdf_path)

        # If still no content, try web_fetch on the URL
        if not result["abstract"] and result["pdf_path"] is None:
            try:
                from biobank_agent.skills.web_fetch import web_fetch

                fetched = web_fetch(url=identifier, max_chars=8000, ctx=ctx)
                if "content" in fetched:
                    result["abstract"] = fetched["content"][:2000]
                    result["title"] = result["title"] or fetched.get("title", "")
            except Exception as exc:
                logger.debug("web_fetch fallback failed: %s", exc)

    elif id_type == "title":
        # Search for the paper
        try:
            from biobank_agent.skills.web_search import web_search

            search_results = web_search(
                query=f"{identifier} scientific paper",
                max_results=5,
                ctx=ctx,
            )
            hits = search_results.get("results", [])
            if hits:
                best = hits[0]
                result["title"] = best.get("title", identifier)
                result["url"] = best.get("url", "")
                result["abstract"] = best.get("snippet", "")

                # Check if we found a DOI in the result
                for hit in hits:
                    doi_match = re.search(
                        r"(10\.\d{4,9}/[^\s&?#]+)",
                        hit.get("url", "") + " " + hit.get("snippet", ""),
                    )
                    if doi_match:
                        doi = doi_match.group(1)
                        result["doi"] = doi
                        meta = _resolve_doi(doi)
                        result.update({k: v for k, v in meta.items() if v})
                        pdf_path = _fetch_pdf_from_doi(doi, papers_dir)
                        if pdf_path:
                            result["pdf_path"] = str(pdf_path)
                        break
        except Exception as exc:
            logger.warning("Title search failed: %s", exc)
            result["error"] = f"Could not find paper by title: {exc}"

    # ------------------------------------------------------------------
    # Extract full text from PDF if we have one
    # ------------------------------------------------------------------

    if result["pdf_path"]:
        try:
            from biobank_agent.skills.read_pdf import read_pdf

            pdf_result = read_pdf(path=result["pdf_path"], max_pages=0, ctx=ctx)
            if "text" in pdf_result and pdf_result["text"]:
                result["full_text"] = pdf_result["text"]
                result["n_pages"] = pdf_result.get("n_pages", 0)
                # Use PDF metadata as fallback for title/authors
                pdf_meta = pdf_result.get("metadata", {})
                if not result["title"] and pdf_meta.get("title"):
                    result["title"] = pdf_meta["title"]
                if not result["authors"] and pdf_meta.get("author"):
                    result["authors"] = [pdf_meta["author"]]
        except Exception as exc:
            logger.warning("PDF text extraction failed: %s", exc)

    # If we have no full text and no abstract, flag it
    if not result["full_text"] and not result["abstract"]:
        result["warning"] = (
            "Could not access the full paper. It may be behind a paywall. "
            "Try accessing through your institution's proxy or library."
        )

    return result
