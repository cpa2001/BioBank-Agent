"""Recall past analysis sessions — cross-session episodic memory."""

from biobank_agent.registry import skill


@skill(
    name="recall_session",
    description="Search past analysis sessions by keyword. Use when the user asks "
                "'what did we find about X last time?' or 'recall our diabetes analysis'. "
                "Returns matching sessions ranked by relevance.",
    parameters={
        "query": {
            "type": "string",
            "description": "Search keywords (e.g., 'diabetes biomarkers', 'E11 model', 'survival I21')",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of sessions to return",
            "default": 5,
        },
    },
    required=["query"],
)
def recall_session(query: str, limit: int = 5, *, ctx=None) -> dict:
    """Search past analysis sessions for relevant findings."""
    if not hasattr(ctx, "memory") or not hasattr(ctx.memory, "sessions"):
        return {"results": [], "message": "Session search not available."}

    results = ctx.memory.sessions.search(query, limit=limit)

    if not results:
        return {
            "results": [],
            "query": query,
            "message": f"No past sessions found matching '{query}'.",
        }

    return {
        "query": query,
        "n_results": len(results),
        "results": results,
    }
