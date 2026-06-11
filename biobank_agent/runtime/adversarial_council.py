"""Adversarial-game planning council: proposer drafts, red-team attacks, referee adjudicates.

The existing council (runtime/planner.py) runs a SYMMETRIC debate — N peers converge. This is
the ASYMMETRIC alternative: a proposer drafts a plan, a red-team tries to break it (missing
data, methodology sins, infeasible steps, hidden assumptions), and a referee decides which
flaws are load-bearing, folds those fixes into a revised plan, and dismisses the rest. The game
is scored — the red-team is rewarded for flaws the referee accepts; the proposer for flaws it
dismisses — so the A/B harness (council_ab.py) can decide whether this beats the symmetric mode.

Pure orchestration over injected role functions, so it is deterministic and unit-testable with
no live model calls; the planner binds the roles to distinct council models behind a flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

BLOCK = "block"
MAJOR = "major"
MINOR = "minor"
_LOAD_BEARING = {BLOCK, MAJOR}


@dataclass
class Flaw:
    severity: str = MINOR        # block | major | minor
    kind: str = ""
    claim: str = ""
    fix: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "kind": self.kind, "claim": self.claim, "fix": self.fix}

    @property
    def load_bearing(self) -> bool:
        return self.severity in _LOAD_BEARING


def _as_flaw(item: Any) -> Flaw:
    if isinstance(item, Flaw):
        return item
    if isinstance(item, dict):
        return Flaw(severity=str(item.get("severity") or MINOR), kind=str(item.get("kind") or ""),
                    claim=str(item.get("claim") or ""), fix=str(item.get("fix") or ""))
    return Flaw(claim=str(item))


def _as_flaws(items: Any) -> list[Flaw]:
    return [_as_flaw(i) for i in (items or [])]


@dataclass
class AdjudicatedPlan:
    plan: str
    addressed: list[Flaw] = field(default_factory=list)   # accepted by referee, folded into plan
    survived: list[Flaw] = field(default_factory=list)    # dismissed by referee (proposer wins)
    notes: str = ""
    rounds: int = 0

    @property
    def game_score(self) -> dict[str, int]:
        """Red-team is credited for accepted load-bearing flaws; proposer for dismissals."""
        return {
            "redteam": sum(1 for f in self.addressed if f.load_bearing),
            "proposer": len(self.survived),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"plan": self.plan, "addressed": [f.to_dict() for f in self.addressed],
                "survived": [f.to_dict() for f in self.survived], "notes": self.notes,
                "rounds": self.rounds, "game_score": self.game_score}


def adversarial_plan(
    objective: str,
    *,
    proposer_fn: Callable[[str], str],
    redteam_fn: Callable[[str, str], Iterable[Any]],
    referee_fn: Callable[[str, str, list[dict[str, str]]], dict[str, Any]],
    rounds: int = 2,
) -> AdjudicatedPlan:
    """Drive proposer -> (red-team -> referee)* up to ``rounds`` times.

    Each round: the red-team critiques the current draft; the referee returns
    ``{"plan", "accepted", "dismissed", "notes"}`` — ``accepted`` flaws are folded into the
    revised plan, ``dismissed`` flaws survive. Stops early once a round accepts no load-bearing
    (block/major) flaw — i.e. the plan has converged. Pure control flow over injected roles."""
    draft = str(proposer_fn(objective) or "")
    addressed: list[Flaw] = []
    survived: list[Flaw] = []
    notes: list[str] = []
    rounds_run = 0

    for _ in range(max(1, int(rounds))):
        rounds_run += 1
        flaws = _as_flaws(redteam_fn(objective, draft))
        verdict = referee_fn(objective, draft, [f.to_dict() for f in flaws]) or {}
        accepted = _as_flaws(verdict.get("accepted"))
        dismissed = _as_flaws(verdict.get("dismissed"))
        addressed.extend(accepted)
        survived.extend(dismissed)
        if verdict.get("notes"):
            notes.append(str(verdict["notes"]))
        revised = verdict.get("plan")
        if revised:
            draft = str(revised)
        if not any(f.load_bearing for f in accepted):
            break

    return AdjudicatedPlan(plan=draft, addressed=addressed, survived=survived,
                           notes=" | ".join(notes), rounds=rounds_run)


__all__ = ["Flaw", "AdjudicatedPlan", "adversarial_plan", "BLOCK", "MAJOR", "MINOR"]
