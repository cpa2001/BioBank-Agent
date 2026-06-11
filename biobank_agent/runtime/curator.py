"""Skill curator: usage-driven tier promotion + auto-harness seeding.

The skill tree (skills/manifest.json, M0) keeps context small by exposing only
``direct`` skills each turn and hiding the rest behind ``skill_search``. But which
skills deserve the ``direct`` tier should be learned from USE, not frozen at design
time. The curator aggregates per-skill usage (calls + success rate) from session
records, recommends promotions (a heavily-used, reliable ``deferred`` skill → ``direct``)
and demotions (a ``direct`` skill that is rarely used or frequently fails → ``deferred``),
applies them to the manifest, and seeds benchmark cases for high-value skills so the
eval harness (runtime/harness.py) grows alongside the toolkit.

Pure decision logic + a single manifest mutation; no model calls, fully unit-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

_OK_STATUSES = {"ok", "success", "succeeded", "done", "completed", "passed"}


@dataclass
class SkillUsageStats:
    skill: str
    calls: int = 0
    successes: int = 0
    failures: int = 0

    @property
    def success_rate(self) -> float:
        return (self.successes / self.calls) if self.calls else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"skill": self.skill, "calls": self.calls, "successes": self.successes,
                "failures": self.failures, "success_rate": round(self.success_rate, 4)}


def _record_field(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def _record_ok(record: Any) -> bool:
    status = str(_record_field(record, "status") or "").lower()
    if status:
        return status in _OK_STATUSES
    success = _record_field(record, "success")
    return bool(success) if success is not None else True   # no signal => assume ok


def collect_usage(records: Iterable[Any]) -> dict[str, SkillUsageStats]:
    """Aggregate per-skill usage from session records (each carries ``skill`` + a
    ``status``/``success`` signal). Records without a skill are ignored."""
    stats: dict[str, SkillUsageStats] = {}
    for record in records or []:
        skill = _record_field(record, "skill")
        if not skill:
            continue
        skill = str(skill)
        st = stats.setdefault(skill, SkillUsageStats(skill))
        st.calls += 1
        if _record_ok(record):
            st.successes += 1
        else:
            st.failures += 1
    return stats


def recommend_curation(
    usage: dict[str, SkillUsageStats],
    exposure_of: Callable[[str], str],
    *,
    min_calls: int = 5,
    promote_success_rate: float = 0.8,
    demote_success_rate: float = 0.4,
    pinned: Callable[[str], bool] | None = None,
    trust_of: Callable[[str], str] | None = None,
) -> dict[str, list[str]]:
    """Decide tier changes from usage (pure). A ``deferred`` skill used >= ``min_calls``
    times at >= ``promote_success_rate`` is promoted to ``direct``; a ``direct`` skill used
    >= ``min_calls`` times but below ``demote_success_rate`` is demoted to ``deferred``.
    Pinned skills are never demoted. ``external`` (third-party-ingested) skills are never
    auto-promoted on usage alone — they require an explicit human enable per corpus."""
    promote: list[str] = []
    demote: list[str] = []
    for skill, st in usage.items():
        tier = exposure_of(skill)
        if st.calls < min_calls:
            continue
        if tier == "deferred" and st.success_rate >= promote_success_rate:
            if trust_of and trust_of(skill) == "external":
                continue
            promote.append(skill)
        elif tier == "direct" and st.success_rate < demote_success_rate:
            if not (pinned and pinned(skill)):
                demote.append(skill)
    return {"promote": sorted(promote), "demote": sorted(demote)}


def apply_curation(recommendations: dict[str, list[str]], manifest_path: str | Path) -> dict[str, list[str]]:
    """Mutate ``manifest.json`` tiers: promote => add to ``tiers.direct``; demote => remove
    from ``tiers.direct`` (so it falls back to ``deferred``). Returns the changes actually
    applied. Never moves a skill into ``hidden`` (that stays a human/curator decision)."""
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    tiers = data.setdefault("tiers", {})
    direct = list(tiers.get("direct") or [])
    hidden = set(tiers.get("hidden") or [])
    applied: dict[str, list[str]] = {"promoted": [], "demoted": []}
    for skill in recommendations.get("promote", []):
        if skill not in direct and skill not in hidden:
            direct.append(skill)
            applied["promoted"].append(skill)
    for skill in recommendations.get("demote", []):
        if skill in direct:
            direct.remove(skill)
            applied["demoted"].append(skill)
    tiers["direct"] = sorted(set(direct))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return applied


def propose_harness_tasks(skills: Iterable[str], *, capability: str = "PYTHON_EXEC") -> list[dict[str, Any]]:
    """Auto-seed a minimal benchmark smoke task per skill (M9 auto-harness) so newly
    promoted / high-value skills gain eval coverage. Each task is a spec dict the
    harness/eval layer can flesh out; deterministic + ordered."""
    tasks: list[dict[str, Any]] = []
    for skill in sorted({str(s) for s in skills if s}):
        tasks.append({
            "id": f"auto::{skill}",
            "skill": skill,
            "capability": capability,
            "objective": f"Exercise the '{skill}' skill on a minimal valid input and produce an artifact.",
            "expectations": {"skill_invoked": skill, "produces_artifact": True},
            "auto_generated": True,
        })
    return tasks


def curate(
    records: Iterable[Any],
    exposure_of: Callable[[str], str],
    manifest_path: str | Path,
    *,
    dry_run: bool = True,
    **thresholds: Any,
) -> dict[str, Any]:
    """End-to-end curation pass: collect usage -> recommend -> (optionally) apply ->
    seed harness tasks for promoted skills. ``dry_run=True`` reports without mutating."""
    usage = collect_usage(records)
    recs = recommend_curation(usage, exposure_of, **thresholds)
    applied: dict[str, list[str]] = {"promoted": [], "demoted": []}
    if not dry_run and (recs["promote"] or recs["demote"]):
        applied = apply_curation(recs, manifest_path)
    harness_tasks = propose_harness_tasks(recs["promote"])
    return {
        "usage": {k: v.to_dict() for k, v in sorted(usage.items())},
        "recommendations": recs,
        "applied": applied,
        "harness_tasks": harness_tasks,
        "dry_run": dry_run,
    }
