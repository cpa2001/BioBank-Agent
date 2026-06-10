"""Pause-and-ask skill for data-format / tool-mismatch situations (issue #3).

When the input data does not match what a standard tool expects (e.g. a VCF
without the FORMAT/AD field a burden test needs, a phenotype table in the wrong
shape), the agent's bad habit is to silently write a SIMPLIFIED replacement
script that only partially does the job — and the user only finds out at the end.
This skill lets the agent instead STOP and surface the mismatch with one concrete
question + options, so the user can fix the data or redirect.

It is signal-only (it does not block inside the model turn): it returns the same
``awaiting_user`` sentinel as ``ask_user``, which the plan loop detects to pause
the run, surface the question, and resume once the user answers.
"""

from __future__ import annotations

from biobank_agent.registry import skill


_CATEGORIES = {"data_format", "tool_mismatch", "ambiguous_input", "missing_path", "missing_tool"}


@skill(
    name="pause_and_ask",
    description=(
        "Pause the run and ask the user ONE concrete question when the data does not "
        "match what a standard tool expects, a required tool/file is missing, or the "
        "input is ambiguous. Use this INSTEAD of silently writing a simplified "
        "replacement script: name the mismatch and propose options. Signal-only — the "
        "step pauses and resumes once the user answers."
    ),
    parameters={
        "question": {"type": "string", "description": "The single concrete question to ask the user."},
        "reason": {"type": "string", "description": "Why you are blocked (e.g. 'VCF has no FORMAT/AD; burden test needs AD').", "default": ""},
        "category": {"type": "string", "description": "data_format | tool_mismatch | ambiguous_input | missing_path | missing_tool", "default": "ambiguous_input"},
        "options": {"type": "string", "description": "Optional comma-separated suggested answers/paths.", "default": ""},
    },
    required=["question"],
)
def pause_and_ask(
    question: str,
    reason: str = "",
    category: str = "ambiguous_input",
    options: str = "",
    *,
    ctx=None,
) -> dict:
    question = str(question or "").strip()
    if not question:
        return {"status": "error", "error": "question is required"}
    category = str(category or "ambiguous_input").strip().lower()
    if category not in _CATEGORIES:
        category = "ambiguous_input"
    option_list = [o.strip() for o in str(options or "").split(",") if o.strip()]
    return {
        "status": "ok",
        "awaiting_user": True,
        "question": question,
        "reason": str(reason or "").strip(),
        "category": category,
        "options": option_list,
        "message": (
            "Mismatch recorded. STOP — do not write a simplified replacement. End your "
            "turn with a brief note that you are waiting for the user; the step resumes "
            "once they answer."
        ),
    }
