"""Web search skill — search the web for scientific papers and biomedical information.

Supports DuckDuckGo (default, no API key), Brave Search, and Serper (Google) backends.
Configure via ``ctx.settings.search_provider`` and ``ctx.settings.search_api_key``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_legacy_duckduckgo_warning():
    """Suppress the legacy package's forced rename RuntimeWarning."""
    import warnings

    original_warn = warnings.warn

    def filtered_warn(message, *args, **kwargs):
        text = str(message)
        if "duckduckgo_search" in text and "ddgs" in text:
            return None
        return original_warn(message, *args, **kwargs)

    warnings.warn = filtered_warn
    try:
        yield
    finally:
        warnings.warn = original_warn


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    """Search via DuckDuckGo (no API key required).

    Handles both the new ``ddgs`` package and the legacy ``duckduckgo_search``.
    """
    import warnings

    # Import: try new package name first, fall back to legacy
    try:
        from ddgs import DDGS
    except ImportError:
        with warnings.catch_warnings(), _suppress_legacy_duckduckgo_warning():
            warnings.simplefilter("ignore", RuntimeWarning)
            from duckduckgo_search import DDGS

    results: list[dict] = []
    try:
        with warnings.catch_warnings(), _suppress_legacy_duckduckgo_warning():
            warnings.simplefilter("ignore", RuntimeWarning)
            ddgs = DDGS()
            raw = ddgs.text(query, max_results=max_results)
            for r in raw:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                })
    except Exception as exc:
        # DuckDuckGo occasionally rate-limits; return whatever we got so far
        logger.warning("DuckDuckGo search interrupted: %s", exc)
        if not results:
            return [{"title": "⚠ Rate limited", "url": "", "snippet": str(exc)}]
    return results


def _search_brave(query: str, max_results: int, api_key: str) -> list[dict]:
    """Search via Brave Search API (requires API key)."""
    import httpx

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": min(max_results, 20)}

    resp = httpx.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    results: list[dict] = []
    for item in data.get("web", {}).get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
        })
    return results[:max_results]


def _search_serper(query: str, max_results: int, api_key: str) -> list[dict]:
    """Search via Serper.dev Google Search API (requires API key)."""
    import httpx

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": min(max_results, 100)}

    resp = httpx.post(url, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    results: list[dict] = []
    for item in data.get("organic", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })
    return results[:max_results]


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

@skill(
    name="web_search",
    description="Search the web for scientific papers, biomedical information, or documentation. "
                "Uses DuckDuckGo by default, upgrades to Brave/Serper when API key is configured.",
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return",
            "default": 10,
        },
    },
    required=["query"],
)
def web_search(query: str, max_results: int = 10, *, ctx=None) -> dict:
    """Execute a web search and return structured results.

    The search backend is selected from ``ctx.settings.search_provider``:

    * ``"duckduckgo"`` (default) — free, no API key needed.
    * ``"brave"`` — requires ``ctx.settings.search_api_key``.
    * ``"serper"`` — requires ``ctx.settings.search_api_key``.

    Returns
    -------
    dict
        ``{"results": [...], "provider": str, "query": str}``
    """
    provider = "duckduckgo"
    api_key = ""

    if ctx is not None and ctx.settings is not None:
        provider = getattr(ctx.settings, "search_provider", "duckduckgo") or "duckduckgo"
        api_key = getattr(ctx.settings, "search_api_key", "") or ""

    provider = provider.lower().strip()

    try:
        if provider == "brave" and api_key:
            results = _search_brave(query, max_results, api_key)
        elif provider == "serper" and api_key:
            results = _search_serper(query, max_results, api_key)
        else:
            # Fallback to DuckDuckGo when key is missing or provider is explicit
            if provider not in ("duckduckgo", "brave", "serper"):
                logger.warning("Unknown search_provider '%s', falling back to DuckDuckGo", provider)
            results = _search_duckduckgo(query, max_results)
    except Exception as exc:
        logger.error("Web search failed (%s): %s", provider, exc)
        return {
            "results": [],
            "provider": provider,
            "query": query,
            "error": f"Search failed: {exc}",
        }

    return {
        "results": results,
        "provider": provider,
        "query": query,
    }
