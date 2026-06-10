"""Method contract: the typed scientific pre/post-condition spec for a skill.

A ``MethodContract`` is the single source of truth shared across the skill-acquisition
engine: the methodology reviewer rejects a contract that trips a consensus statistical or
omics sin *before* any code is generated, paper-synthesis embeds it as the synthesized
skill's docstring + proposal summary, and external ingestion carries it per skill. One
dataclass, imported everywhere — never forked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MethodContract:
    """Declared scientific contract for an analysis method.

    ``postconditions`` are the observable properties an artifact must satisfy (checked by
    ``methodology.check_artifact`` on a JSON summary, without importing the heavy library).
    """

    name: str = ""
    summary: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    statistical_assumptions: list[str] = field(default_factory=list)
    citation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MethodContract":
        if not isinstance(data, dict):
            raise TypeError("MethodContract.from_dict expects a dict")
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def is_empty(self) -> bool:
        """True when the contract declares no scientific content (only a name/citation)."""
        return not any([
            self.inputs, self.outputs, self.preconditions,
            self.postconditions, self.statistical_assumptions,
        ])

    def as_text(self) -> str:
        """Flatten the contract to a single string for the methodology reviewer's narrative
        scan (so a contract can be reviewed before any result exists)."""
        parts = [self.summary]
        for label, items in (
            ("inputs", self.inputs), ("outputs", self.outputs),
            ("preconditions", self.preconditions), ("postconditions", self.postconditions),
            ("assumptions", self.statistical_assumptions),
        ):
            if items:
                parts.append(f"{label}: " + "; ".join(str(i) for i in items))
        return "\n".join(p for p in parts if p)


__all__ = ["MethodContract"]
