"""Search UKB field catalogue by keyword or field ID."""

from __future__ import annotations

import re
from collections import defaultdict

from biobank_agent.registry import skill


_FIELD_QUERY_SYNONYMS = {
    "bmi": ["BMI", "body mass index"],
    "hba1c": ["HbA1c", "glycated haemoglobin", "HBA1C"],
    "t2d": ["Type 2 Diabetes", "diabetes", "E11"],
    "diabetes": ["diabetes", "Type 2 Diabetes", "diabetes diagnosed"],
    "glucose": ["glucose", "blood glucose"],
    "bp": ["blood pressure", "systolic blood pressure", "diastolic blood pressure"],
    "pressure": ["blood pressure", "systolic blood pressure", "diastolic blood pressure"],
    "longitudinal": ["follow-up", "repeat assessment", "instance", "date", "age"],
    "repeated": ["repeat assessment", "follow-up", "instance"],
    "cholesterol": ["cholesterol", "LDL cholesterol", "HDL cholesterol"],
    "triglycerides": ["triglycerides"],
}

_STOPWORDS = {
    "and", "or", "the", "with", "from", "for", "into", "over", "time",
    "type", "repeated", "measures", "measure", "longitudinal", "trajectory",
    "forecast", "prediction", "predict", "disease", "progression", "ukb",
    "uk", "biobank",
}


def _expanded_field_queries(query: str, limit: int) -> list[str]:
    """Expand a human biomedical search phrase into catalogue-friendly probes."""
    clean = " ".join(str(query or "").replace("/", " ").split())
    if not clean:
        return []

    queries: list[str] = [clean]
    lower = clean.lower()
    tokens = [t for t in re.findall(r"[A-Za-z0-9]+", lower) if len(t) > 1]

    for token in tokens:
        for synonym in _FIELD_QUERY_SYNONYMS.get(token, []):
            queries.append(synonym)

    # Phrase-level expansions catch terms split into multiple tokens.
    phrase_synonyms = {
        "type 2 diabetes": ["Type 2 Diabetes", "diabetes diagnosed", "E11"],
        "blood pressure": ["blood pressure", "systolic blood pressure", "diastolic blood pressure"],
        "body mass index": ["body mass index", "BMI"],
        "glycated haemoglobin": ["glycated haemoglobin", "HbA1c"],
    }
    for phrase, synonyms in phrase_synonyms.items():
        if phrase in lower:
            queries.extend(synonyms)

    for token in tokens:
        if token not in _STOPWORDS and not token.isdigit():
            queries.append(token)

    seen: set[str] = set()
    out: list[str] = []
    for item in queries:
        item = " ".join(str(item).split())
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max(8, min(32, limit * 3)):
            break
    return out


def _field_relevance_score(row: dict, query: str, matched_queries: list[str]) -> int:
    title = str(row.get("title", "")).lower()
    desc = str(row.get("description", "")).lower()
    haystack = f"{title} {desc}"
    score = 0
    for matched in matched_queries:
        m = matched.lower()
        if not m:
            continue
        if m == title:
            score += 20
        elif m in title:
            score += 12
        elif m in haystack:
            score += 6
    for token in re.findall(r"[a-z0-9]+", query.lower()):
        if token in _STOPWORDS or len(token) < 3:
            continue
        if token in title:
            score += 3
        elif token in haystack:
            score += 1
    source = str(row.get("data_source", "")).lower()
    if source == "parquet":
        score += 3
    elif source == "csv":
        score += 1
    return score


@skill(
    name="field_search",
    description="Search the UK Biobank field catalogue by keyword or field ID. "
                "Returns matching field definitions with their IDs, names, categories, "
                "value types, and units. Use this to discover which fields are available.",
    parameters={
        "query": {
            "type": "string",
            "description": "Search keyword (e.g. 'glucose', 'smoking', 'cholesterol') "
                           "or field ID (e.g. '30740')",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum results to return (default 20)",
            "default": 20,
        },
    },
    required=["query"],
)
def field_search(query: str = "", limit: int = 20, *, ctx=None) -> dict:
    query = str(query or "").strip()
    if not query:
        return {
            "query": query,
            "results": [],
            "total": 0,
            "error": "Missing query",
            "message": "Provide a field ID or catalogue keyword, for example '30740' or 'glucose'.",
        }
    if ctx is None or not hasattr(ctx, "catalog"):
        return {
            "query": query,
            "results": [],
            "total": 0,
            "error": "Field catalogue unavailable",
        }
    catalog = ctx.catalog

    # Check if query is a field ID
    if query.isdigit():
        info = catalog.field_info(query)
        if info:
            cat_name = catalog.category_name(info.get("category_id", ""))
            return {
                "results": [{
                    "field_id": query,
                    "title": info["title"],
                    "category": cat_name,
                    "value_type": info.get("value_type", ""),
                    "units": info.get("units", ""),
                }],
                "total": 1,
            }
        return {"results": [], "total": 0, "message": f"Field {query} not found"}

    # Keyword search. UKB Showcase search is exact-substring oriented, so a
    # human phrase like "longitudinal repeated measures BMI HbA1c..." needs
    # catalogue-friendly fallback probes rather than being treated as zero data.
    by_field: dict[str, dict] = {}
    matched_by_field: dict[str, list[str]] = defaultdict(list)
    searched_queries: list[str] = []
    for probe in _expanded_field_queries(query, limit):
        try:
            search_limit = limit if probe.lower() == query.lower() else max(limit, 20)
            results = catalog.search(probe, limit=search_limit)
        except Exception:
            results = []
        searched_queries.append(probe)
        for r in results:
            fid = str(r.get("field_id", ""))
            if not fid:
                continue
            if fid not in by_field:
                cat_name = catalog.category_name(r.get("category_id", ""))
                dm = getattr(ctx, "dm", None)
                source = dm.field_source(fid) if dm is not None else "unknown"
                by_field[fid] = {
                    "field_id": fid,
                    "title": r.get("title", ""),
                    "category": cat_name,
                    "value_type": r.get("value_type", ""),
                    "units": r.get("units", ""),
                    "data_source": source,
                }
            matched_by_field[fid].append(probe)
        if len(by_field) >= limit and probe.lower() == query.lower():
            break

    formatted = []
    for fid, row in by_field.items():
        item = dict(row)
        item["matched_queries"] = matched_by_field.get(fid, [])
        item["relevance_score"] = _field_relevance_score(item, query, item["matched_queries"])
        formatted.append(item)
    formatted.sort(key=lambda r: (-r.get("relevance_score", 0), str(r.get("field_id", ""))))
    formatted = formatted[:limit]

    warnings = []
    if not formatted:
        warnings.append(
            "No catalogue fields matched the original phrase or expanded biomedical terms; "
            "try a specific field name such as BMI, HbA1c, glucose, or systolic blood pressure."
        )

    return {
        "query": query,
        "results": formatted,
        "total": len(formatted),
        "status": "READY" if formatted else "PARTIAL",
        "matched_queries": sorted({q for item in formatted for q in item.get("matched_queries", [])}),
        "searched_queries": searched_queries,
        "suggested_queries": ["BMI", "HbA1c", "glucose", "systolic blood pressure", "diabetes diagnosed"],
        "warnings": warnings,
        "requires_repair": not bool(formatted),
    }
