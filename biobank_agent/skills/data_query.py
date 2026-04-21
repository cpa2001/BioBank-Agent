"""Search UKB field catalogue by keyword or field ID."""

from biobank_agent.registry import skill


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
def field_search(query: str, limit: int = 20, *, ctx=None) -> dict:
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

    # Keyword search
    results = catalog.search(query, limit=limit)
    formatted = []
    for r in results:
        cat_name = catalog.category_name(r.get("category_id", ""))
        # Check if field is in parquet (fast) or CSV (slow)
        source = ctx.dm.field_source(r["field_id"])
        formatted.append({
            "field_id": r["field_id"],
            "title": r["title"],
            "category": cat_name,
            "value_type": r.get("value_type", ""),
            "units": r.get("units", ""),
            "data_source": source,
        })

    return {
        "query": query,
        "results": formatted,
        "total": len(formatted),
    }
