"""Internal reasoning tool — gives the LLM a scratchpad.

Enhanced with optional Tree-of-Thought: when the reasoning contains
branching decisions (e.g., "which approach should I use?"), the think
tool can activate multi-path exploration to evaluate alternatives
before committing.
"""

from biobank_agent.registry import skill


@skill(
    name="think",
    description="Use this to reason through a complex problem step by step before taking action. "
                "Your reasoning is logged but not shown to the user. "
                "For questions with multiple valid approaches, this tool can explore "
                "alternative paths and recommend the best one.",
    parameters={
        "reasoning": {
            "type": "string",
            "description": "Your internal step-by-step reasoning",
        },
    },
)
def think(reasoning: str, *, ctx=None) -> dict:
    """Process internal reasoning, optionally with Tree-of-Thought exploration."""
    # Check if ToT is enabled and the reasoning is a branching question
    enable_tot = False
    if ctx and hasattr(ctx, "settings"):
        enable_tot = getattr(ctx.settings, "enable_tot", False)

    if enable_tot:
        from biobank_agent.reasoning import is_branching_question, ThoughtTree
        if is_branching_question(reasoning):
            try:
                from biobank_agent.llm import LLMClient
                # Use the existing LLM client via the agent's settings
                llm = LLMClient(
                    base_url=ctx.settings.llm_base_url,
                    api_key=ctx.settings.llm_api_key,
                    model=ctx.settings.llm_model,
                )
                tree = ThoughtTree(llm, max_branches=3)
                context = ctx.state.context_summary() if hasattr(ctx, "state") else ""

                result = tree.explore(reasoning, context=context)
                return {
                    "acknowledged": True,
                    "tree_of_thought": True,
                    "paths": [
                        {
                            "approach": p.approach,
                            "score": p.score,
                            "steps": p.steps[:5],
                            "rationale": p.rationale[:200],
                        }
                        for p in result.paths[:3]
                    ],
                    "recommended": result.recommended.approach if result.recommended else None,
                    "consensus": result.consensus,
                }
            except Exception:
                pass  # Fall through to simple acknowledgement

    return {"acknowledged": True}
