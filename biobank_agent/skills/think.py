"""Internal reasoning tool — gives the LLM a scratchpad."""

from biobank_agent.registry import skill


@skill(
    name="think",
    description="Use this to reason through a complex problem step by step before taking action. "
                "Your reasoning is logged but not shown to the user.",
    parameters={
        "reasoning": {
            "type": "string",
            "description": "Your internal step-by-step reasoning",
        },
    },
)
def think(reasoning: str, *, ctx=None) -> dict:
    return {"acknowledged": True}
