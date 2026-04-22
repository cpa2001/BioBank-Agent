"""Deep research skill — multi-source literature + biobank cross-reference.

Searches the web for recent literature on a topic, fetches abstracts,
cross-references with the biobank field catalogue, and compiles
a cited research brief.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _search_literature(topic: str, max_sources: int, ctx: Any) -> list[dict]:
    """Search for recent literature on *topic*."""
    bank_name = ctx.settings.biobank_name if ctx and hasattr(ctx, "settings") else "Biobank"
    try:
        from biobank_agent.skills.web_search import web_search

        # Two search passes: general + biobank-specific
        general = web_search(
            query=f"{topic} scientific study",
            max_results=max_sources,
            ctx=ctx,
        )
        biobank = web_search(
            query=f"{topic} {bank_name}",
            max_results=max(3, max_sources // 3),
            ctx=ctx,
        )

        all_results = general.get("results", []) + biobank.get("results", [])

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique: list[dict] = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(r)
        return unique[:max_sources]

    except Exception as exc:
        logger.warning("Literature search failed: %s", exc)
        return []


def _fetch_abstracts(sources: list[dict], max_fetch: int, ctx: Any) -> list[dict]:
    """Fetch content from top sources to get richer abstracts."""
    try:
        from biobank_agent.skills.web_fetch import web_fetch
    except ImportError:
        return sources

    enriched = []
    fetched_count = 0
    for source in sources:
        entry = dict(source)
        url = source.get("url", "")
        if url and fetched_count < max_fetch:
            try:
                result = web_fetch(url=url, max_chars=2000, ctx=ctx)
                if "content" in result and not result.get("error"):
                    entry["abstract"] = result["content"]
                    entry["title"] = result.get("title") or entry.get("title", "")
                    fetched_count += 1
            except Exception as exc:
                logger.debug("Failed to fetch %s: %s", url, exc)
        enriched.append(entry)
    return enriched


def _cross_reference_biobank_fields(topic: str, ctx: Any) -> list[dict]:
    """Search the biobank field catalogue for fields relevant to *topic*."""
    if ctx is None or not hasattr(ctx, "catalog"):
        return []

    relevant_fields: list[dict] = []
    try:
        # Split topic into searchable keywords
        keywords = [w.strip() for w in topic.split() if len(w.strip()) > 3]
        # Also search the full topic
        keywords = [topic] + keywords[:5]

        seen_ids: set[str] = set()
        for keyword in keywords:
            try:
                results = ctx.catalog.search(keyword, limit=10)
                for r in results:
                    fid = r.get("field_id", "")
                    if fid and fid not in seen_ids:
                        seen_ids.add(fid)
                        cat_name = ""
                        try:
                            cat_name = ctx.catalog.category_name(r.get("category_id", ""))
                        except Exception:
                            pass
                        relevant_fields.append({
                            "field_id": fid,
                            "title": r.get("title", ""),
                            "category": cat_name,
                            "value_type": r.get("value_type", ""),
                        })
            except Exception:
                continue

    except Exception as exc:
        logger.debug("Biobank field cross-reference failed: %s", exc)

    return relevant_fields[:30]  # cap at 30


def _compile_brief(
    topic: str,
    sources: list[dict],
    biobank_fields: list[dict],
    bank_name: str = "Biobank",
) -> str:
    """Compile a markdown research brief."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Research Brief: {topic}",
        f"**Generated**: {now}",
        f"**Sources reviewed**: {len(sources)}",
        f"**{bank_name} fields identified**: {len(biobank_fields)}",
        "",
        "---",
        "",
        "## Literature Overview",
        "",
    ]

    if sources:
        for i, s in enumerate(sources, 1):
            title = s.get("title", "Untitled")
            url = s.get("url", "")
            snippet = s.get("snippet", s.get("abstract", ""))
            # Truncate snippet
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            lines.append(f"### [{i}] {title}")
            if url:
                lines.append(f"**URL**: {url}")
            lines.append(f"\n{snippet}\n")
    else:
        lines.append("*No sources found. Try refining the search topic.*\n")

    lines.extend([
        "---",
        "",
        f"## {bank_name} Relevant Fields",
        "",
    ])

    if biobank_fields:
        lines.append("| Field ID | Title | Category | Type |")
        lines.append("|----------|-------|----------|------|")
        for f in biobank_fields:
            lines.append(
                f"| {f['field_id']} | {f['title']} | {f['category']} | {f['value_type']} |"
            )
        lines.append("")
    else:
        lines.append(f"*No directly matching {bank_name} fields found. Manual catalogue review recommended.*\n")

    lines.extend([
        "---",
        "",
        "## Synthesis & Next Steps",
        "",
        f"Based on the literature and available {bank_name} data, consider:",
        "",
        "1. **Key findings across sources**: [Summarise common themes]",
        "2. **Gaps in current research**: [What questions remain unanswered?]",
        f"3. **{bank_name} data availability**: [Can the identified fields answer remaining questions?]",
        "4. **Recommended analyses**: [Specific analyses to run with the Biobank Agent]",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

@skill(
    name="deep_research",
    description="Conduct multi-source research on a biomedical topic. Searches literature, "
                "cross-references with biobank data, and produces a cited research brief.",
    parameters={
        "topic": {"type": "string", "description": "Research topic"},
        "max_sources": {
            "type": "integer",
            "description": "Maximum number of sources to include",
            "default": 10,
        },
    },
    required=["topic"],
)
def deep_research(topic: str, max_sources: int = 10, *, ctx=None) -> dict:
    """Conduct multi-source research and produce a cited brief.

    Pipeline:

    1. Search the web for recent papers and resources on *topic*.
    2. Fetch abstracts/content from the top results.
    3. Cross-reference with the biobank field catalogue.
    4. Compile a structured research brief with citations.

    Returns
    -------
    dict
        ``{"topic": ..., "sources": [...], "biobank_relevant_fields": [...],
           "brief": str}``
    """
    max_sources = max(1, min(max_sources, 30))  # clamp
    bank_name = ctx.settings.biobank_name if ctx and hasattr(ctx, "settings") else "Biobank"

    # Step 1: Search literature
    sources = _search_literature(topic, max_sources, ctx)

    # Step 2: Fetch abstracts for top results (limit network calls)
    max_fetch = min(5, len(sources))
    if sources:
        sources = _fetch_abstracts(sources, max_fetch, ctx)

    # Step 3: Cross-reference with biobank catalogue
    biobank_fields = _cross_reference_biobank_fields(topic, ctx)

    # Step 4: Compile brief
    brief = _compile_brief(topic, sources, biobank_fields, bank_name=bank_name)

    # Save brief to report directory
    brief_path: str | None = None
    try:
        if ctx is not None and hasattr(ctx, "report_dir"):
            research_dir = Path(ctx.report_dir) / "research"
        else:
            research_dir = Path("./research")
        research_dir.mkdir(parents=True, exist_ok=True)

        safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)
        safe_topic = safe_topic.strip().replace(" ", "_")[:60]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_{safe_topic}_{timestamp}.md"

        file_path = research_dir / filename
        file_path.write_text(brief, encoding="utf-8")
        brief_path = str(file_path)
    except Exception as exc:
        logger.warning("Could not save research brief: %s", exc)

    return {
        "topic": topic,
        "sources": [
            {
                "title": s.get("title", ""),
                "url": s.get("url", ""),
                "snippet": s.get("snippet", "")[:200],
            }
            for s in sources
        ],
        "n_sources": len(sources),
        "biobank_relevant_fields": biobank_fields,
        "n_biobank_fields": len(biobank_fields),
        "brief": brief,
        "brief_path": brief_path,
    }
