"""Web fetch skill — retrieve and extract readable text from a URL.

Converts HTML to markdown, with special handling for arXiv and PubMed pages.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML extraction helpers
# ---------------------------------------------------------------------------

def _extract_title(html: str) -> str:
    """Pull <title> text from raw HTML."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_arxiv_abstract(html: str) -> str | None:
    """Extract the abstract block from an arXiv abs page."""
    m = re.search(
        r'<blockquote\s+class="abstract[^"]*">(.*?)</blockquote>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        # Remove leading "Abstract:" label
        text = re.sub(r"^Abstract:\s*", "", text, flags=re.IGNORECASE)
        return text
    return None


def _extract_pubmed_abstract(html: str) -> str | None:
    """Extract the abstract section from a PubMed page."""
    m = re.search(
        r'<div[^>]*class="abstract-content[^"]*"[^>]*>(.*?)</div>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        return text
    return None


def _html_to_text(html: str) -> str:
    """Convert HTML to readable text/markdown via html2text."""
    try:
        import html2text

        converter = html2text.HTML2Text()
        converter.body_width = 0           # no wrapping
        converter.ignore_links = False
        converter.ignore_images = True
        converter.ignore_emphasis = False
        converter.single_line_break = True
        return converter.handle(html)
    except ImportError:
        # Fallback: naive tag-strip
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

@skill(
    name="web_fetch",
    description="Fetch and extract text content from a URL. Converts HTML to readable markdown. "
                "Useful for reading documentation, paper abstracts, and web resources.",
    parameters={
        "url": {"type": "string", "description": "URL to fetch"},
        "max_chars": {
            "type": "integer",
            "description": "Maximum characters to return",
            "default": 8000,
        },
    },
    required=["url"],
)
def web_fetch(url: str, max_chars: int = 8000, *, ctx=None) -> dict:
    """Fetch *url*, convert to readable text, and return truncated content.

    Special extraction rules apply for arXiv and PubMed URLs so that the
    abstract is surfaced cleanly even if the page is messy.

    Returns
    -------
    dict
        ``{"url": ..., "title": ..., "content": ..., "content_length": int,
           "truncated": bool}``
    """
    import httpx

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; BiobankAgent/0.1; "
            "+https://github.com/biobank-agent)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        resp = httpx.get(url, follow_redirects=True, timeout=30, headers=headers)
        resp.raise_for_status()
    except httpx.TimeoutException:
        return {"url": url, "error": "Request timed out after 30 seconds"}
    except httpx.HTTPStatusError as exc:
        return {"url": url, "error": f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}"}
    except Exception as exc:
        return {"url": url, "error": f"Fetch failed: {exc}"}

    content_type = resp.headers.get("content-type", "")
    raw_html = resp.text

    # --- Title ---
    title = _extract_title(raw_html)

    # --- Content extraction (site-specific) ---
    content: str | None = None

    # arXiv abstract pages
    if "arxiv.org" in url:
        content = _extract_arxiv_abstract(raw_html)
        if content:
            content = f"## arXiv Abstract\n\n{content}"

    # PubMed
    if content is None and "pubmed" in url.lower():
        content = _extract_pubmed_abstract(raw_html)
        if content:
            content = f"## PubMed Abstract\n\n{content}"

    # Generic HTML → text fallback
    if content is None:
        if "html" in content_type or raw_html.strip().startswith("<"):
            content = _html_to_text(raw_html)
        else:
            # Plain text / JSON / etc.
            content = raw_html

    # --- Truncation ---
    full_length = len(content)
    truncated = full_length > max_chars
    if truncated:
        content = content[:max_chars] + "\n\n[... truncated ...]"

    return {
        "url": url,
        "title": title,
        "content": content,
        "content_length": full_length,
        "truncated": truncated,
    }
