"""PII filter façade.

Single-import ``filter_pii(payload)`` so domain-side callers don't
have to know about ``biobank_agent.core.events.scrub_pii``.
"""

from __future__ import annotations

from typing import Any

from biobank_agent.core.events import scrub_pii


def filter_pii(payload: Any) -> Any:
    """Strip PII / PHI from ``payload``, recursively."""
    if isinstance(payload, dict):
        return scrub_pii(payload)
    return payload


__all__ = ["filter_pii"]
