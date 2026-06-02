"""Approval policy for tool execution.

Biobank-specific 4-tier model that **replaces** gemini's
``PLAN/AUTO_EDIT/YOLO`` (which are coding-agent abstractions). The v3
plan §5.3 spelled out the right axis for biomedical workflows:

    READ_DATA         - safe by default, readonly DB queries
    EXPORT_AGGREGATE  - aggregate stats (k-anon enforced)
    EXPORT_PII        - per-subject data — explicit user approval
    CALL_REVIEWER     - external LLM reviewer (cost gate)

The policy maps each tool's ``required_capabilities`` set to a
``Decision`` (ALLOW / ASK_USER / DENY) using the active
``ApprovalProfile``.

M3 will plug ``policy_engine.py`` (k-anonymity, min-cell, PHI filter)
into the ``ASK_USER`` branch so soft-failure compliance reviews
happen in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .protocol import Capability, ToolHandler


class Decision(str, Enum):
    ALLOW = "allow"
    ASK_USER = "ask_user"
    DENY = "deny"


@dataclass
class ApprovalProfile:
    """Per-capability decision rules.

    Each ``allow`` set lists capabilities that *do not* prompt; each
    ``ask`` set lists capabilities that prompt before execution; the
    ``deny`` set short-circuits with an error.
    """

    name: str = "default"
    allow: frozenset[Capability] = field(
        default_factory=lambda: frozenset({Capability.READ_DATA})
    )
    ask: frozenset[Capability] = field(
        default_factory=lambda: frozenset({
            Capability.WRITE_REPORTS,
            Capability.NETWORK,
            Capability.EXPORT_AGGREGATE,
            Capability.MUTATE_MEMORY,
            Capability.CALL_REVIEWER,
        })
    )
    deny: frozenset[Capability] = field(
        default_factory=lambda: frozenset({
            Capability.EXPORT_PII,
            Capability.SHELL_EXEC,
        })
    )
    # Mutating tools always escalate even if the capability is "allow".
    escalate_mutating: bool = True


def builtin_profile(level: str = "default") -> ApprovalProfile:
    """Return one of the named profiles.

    Levels:
        "read_only" / "readonly"     - only READ_DATA passes.
        "plan"                       - read-only plus approval for edits, shell,
                                       network, memory mutation, reviewers.
        "ask_before_edits"           - like plan, but files/workspace writes are
                                       asked explicitly while read-only tools pass.
        "workspace_write"            - allow workspace writes, ask for shell/network.
        "accept_edits"               - allow edits after approval prompts; shell
                                       and network still ask.
        "full_auto" / "yolo"         - allow everything; explicit opt-in only.
    """
    mode = str(level or "default").strip().lower().replace("-", "_")
    if mode in {"read_only", "readonly"}:
        return ApprovalProfile(
            name="read_only",
            allow=frozenset({Capability.READ_DATA}),
            ask=frozenset(),
            deny=frozenset({c for c in Capability if c is not Capability.READ_DATA}),
            escalate_mutating=True,
        )
    if mode == "plan":
        return ApprovalProfile(
            name="plan",
            allow=frozenset({Capability.READ_DATA}),
            ask=frozenset({
                Capability.NETWORK,
                Capability.EXPORT_AGGREGATE,
                Capability.MUTATE_MEMORY,
                Capability.CALL_REVIEWER,
            }),
            deny=frozenset({
                Capability.EXPORT_PII,
                Capability.WRITE_REPORTS,
                Capability.SHELL_EXEC,
            }),
            escalate_mutating=True,
        )
    if mode == "ask_before_edits":
        return ApprovalProfile(
            name="ask_before_edits",
            allow=frozenset({Capability.READ_DATA}),
            ask=frozenset({
                Capability.WRITE_REPORTS,
                Capability.NETWORK,
                Capability.EXPORT_AGGREGATE,
                Capability.MUTATE_MEMORY,
                Capability.CALL_REVIEWER,
                Capability.SHELL_EXEC,
            }),
            deny=frozenset({Capability.EXPORT_PII}),
            escalate_mutating=True,
        )
    if mode == "workspace_write":
        return ApprovalProfile(
            name="workspace_write",
            allow=frozenset({Capability.READ_DATA, Capability.WRITE_REPORTS}),
            ask=frozenset({
                Capability.NETWORK,
                Capability.EXPORT_AGGREGATE,
                Capability.MUTATE_MEMORY,
                Capability.CALL_REVIEWER,
                Capability.SHELL_EXEC,
            }),
            deny=frozenset({Capability.EXPORT_PII}),
            escalate_mutating=True,
        )
    if mode == "accept_edits":
        return ApprovalProfile(
            name="accept_edits",
            allow=frozenset({Capability.READ_DATA, Capability.WRITE_REPORTS}),
            ask=frozenset({
                Capability.NETWORK,
                Capability.EXPORT_AGGREGATE,
                Capability.MUTATE_MEMORY,
                Capability.CALL_REVIEWER,
                Capability.SHELL_EXEC,
            }),
            deny=frozenset({Capability.EXPORT_PII}),
            escalate_mutating=False,
        )
    if mode in {"full_auto", "yolo"}:
        return ApprovalProfile(
            name="full_auto",
            allow=frozenset(c for c in Capability),
            ask=frozenset(),
            deny=frozenset(),
            escalate_mutating=False,
        )
    if mode == "permissive":
        return ApprovalProfile(
            name="permissive",
            allow=frozenset(
                c for c in Capability
                if c not in {Capability.EXPORT_PII, Capability.SHELL_EXEC}
            ),
            ask=frozenset(),
            deny=frozenset({Capability.EXPORT_PII, Capability.SHELL_EXEC}),
            escalate_mutating=True,
        )
    if level == "yolo":
        return ApprovalProfile(
            name="yolo",
            allow=frozenset(c for c in Capability),
            ask=frozenset(),
            deny=frozenset(),
            escalate_mutating=False,
        )
    return ApprovalProfile(name="default")


@dataclass
class ApprovalOutcome:
    decision: Decision
    rationale: str = ""
    capability: Optional[Capability] = None
    profile: str = ""


class ApprovalPolicy:
    """Maps (handler, args) → ApprovalOutcome.

    The policy decides only — it does not actually prompt the user;
    that's the scheduler's job (M2.2). Keeping this pure makes it
    cheap to unit test and reusable from non-interactive contexts
    (eval harness, sub-agents).
    """

    def __init__(self, profile: Optional[ApprovalProfile] = None) -> None:
        self.profile = profile or ApprovalProfile()

    def decide(
        self,
        handler: ToolHandler,
        args: dict,
    ) -> ApprovalOutcome:
        caps = handler.required_capabilities()
        # DENY wins.
        for cap in caps:
            if cap in self.profile.deny:
                return ApprovalOutcome(
                    decision=Decision.DENY,
                    rationale=f"capability {cap.value} is denied by profile {self.profile.name!r}",
                    capability=cap,
                    profile=self.profile.name,
                )
        # ASK_USER on first capability matching.
        for cap in caps:
            if cap in self.profile.ask:
                return ApprovalOutcome(
                    decision=Decision.ASK_USER,
                    rationale=f"capability {cap.value} requires explicit approval",
                    capability=cap,
                    profile=self.profile.name,
                )
        # Mutating tools escalate when configured.
        if handler.is_mutating and self.profile.escalate_mutating:
            return ApprovalOutcome(
                decision=Decision.ASK_USER,
                rationale="mutating tool — explicit approval required",
                profile=self.profile.name,
            )
        # Otherwise allow.
        return ApprovalOutcome(
            decision=Decision.ALLOW,
            rationale="capabilities satisfied by profile",
            profile=self.profile.name,
        )


__all__ = [
    "Decision",
    "ApprovalProfile",
    "ApprovalOutcome",
    "ApprovalPolicy",
    "builtin_profile",
]
