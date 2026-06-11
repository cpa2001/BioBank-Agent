"""Ingest external skill libraries (GitHub corpora) as deferred KNOWLEDGE skills.

v1 is metadata-only and SAFE BY CONSTRUCTION: it reads each ``SKILL.md``'s frontmatter +
instructions and registers a knowledge skill whose "execution" returns that guidance — the
third-party repo's CODE is never written to disk or ``exec``'d. Ingested skills are tagged
``trust='external'`` (so the curator never auto-promotes them, M13) and filed into the skill
tree (M10). Raw ``.py`` ingestion and script-backed adapters (which need a hash-pinned runner)
are deferred to M14.2.

Pure parsing/registration with injected deps so it is unit-testable offline; the live
``@skill ingest_github_skills`` (skills/ingest_skills.py) wires the clone + the real registry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from biobank_agent.registry import get_registry
from biobank_agent.skills import manifest, skill_tree

EXTERNAL = "external"


@dataclass
class DiscoveredSkill:
    """One skill found in a corpus: its metadata + guidance text (no executable code)."""

    name: str
    description: str = ""
    instructions: str = ""
    kind: str = "skill_md"
    path: str = ""


@dataclass
class SkillPackage:
    """Provenance for an ingested corpus, persisted as a ``SKILL_PACKAGE.json`` sidecar."""

    name: str
    source_url: str = ""
    commit: str = ""
    license: str = ""
    trust: str = EXTERNAL
    skills: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "source_url": self.source_url, "commit": self.commit,
                "license": self.license, "trust": self.trust, "skills": list(self.skills)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillPackage":
        return cls(
            name=str(data.get("name") or ""),
            source_url=str(data.get("source_url") or ""),
            commit=str(data.get("commit") or ""),
            license=str(data.get("license") or ""),
            trust=str(data.get("trust") or EXTERNAL),
            skills=[str(s) for s in (data.get("skills") or [])],
        )


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(name).strip().lower()).strip("_")


_FRONTMATTER_RE = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def parse_skill_md(text: str) -> dict[str, str]:
    """Parse a SKILL.md: simple ``key: value`` YAML frontmatter + markdown body (instructions).

    Deliberately dependency-free (no yaml): SKILL.md frontmatter is flat key/value. The body
    after the closing ``---`` becomes ``instructions``."""
    raw = text or ""
    meta: dict[str, str] = {}
    body = raw
    m = _FRONTMATTER_RE.match(raw)
    if m:
        front, body = m.group(1), m.group(2)
        for line in front.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    meta["instructions"] = body.strip()
    return meta


def discover_skill_md(root: str | Path) -> list[DiscoveredSkill]:
    """Find every ``SKILL.md`` under ``root`` and parse it into a DiscoveredSkill (sorted)."""
    base = Path(root)
    found: list[DiscoveredSkill] = []
    if not base.exists():
        return found
    for md in sorted(base.rglob("SKILL.md")):
        meta = parse_skill_md(md.read_text(encoding="utf-8", errors="replace"))
        name = _slug(meta.get("name") or md.parent.name)
        if not name:
            continue
        found.append(DiscoveredSkill(name=name, description=meta.get("description", ""),
                                     instructions=meta.get("instructions", ""), path=str(md)))
    return found


def make_knowledge_skill(skill: DiscoveredSkill) -> tuple[Callable[..., dict], dict[str, Any]]:
    """Build a (callable, schema) pair for a knowledge skill. Invoking it returns the SKILL.md
    guidance — no third-party code runs, ever."""
    name, description, instructions = skill.name, skill.description, skill.instructions
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": (description or name)[:1024],
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    def _knowledge(*, ctx=None) -> dict:
        return {"status": "guidance", "skill": name, "description": description,
                "instructions": instructions}

    _knowledge.__name__ = f"knowledge_{name}"
    return _knowledge, schema


def ingest_knowledge_corpus(
    sources: Iterable[DiscoveredSkill],
    package: SkillPackage,
    *,
    register_fn: Callable[[str, Callable[..., dict], dict], None] | None = None,
    classify_fn: Callable[[str, str], str] | None = None,
    trust_fn: Callable[[Iterable[str]], None] | None = None,
) -> dict[str, Any]:
    """Register each discovered skill as a deferred knowledge skill, classify it into the tree,
    and tag the whole batch ``trust='external'``. Pure orchestration over injected deps; never
    writes or executes third-party code. Returns a status dict + the populated SkillPackage."""
    register = register_fn or get_registry().register
    classify = classify_fn or (lambda n, d: skill_tree.classify_skill(n, d))
    tag_trust = trust_fn or manifest.register_external_trust

    ingested: list[dict[str, str]] = []
    seen: set[str] = set()
    for src in sources:
        if not src.name or src.name in seen:
            continue
        seen.add(src.name)
        func, schema = make_knowledge_skill(src)
        register(src.name, func, schema)
        leaf = classify(src.name, src.description or src.name)
        ingested.append({"name": src.name, "leaf": leaf})

    names = [item["name"] for item in ingested]
    tag_trust(names)              # external -> curator never auto-promotes (M13)
    package.skills = names
    return {"status": "ingested", "package": package.to_dict(),
            "skills": ingested, "count": len(ingested)}


__all__ = ["DiscoveredSkill", "SkillPackage", "parse_skill_md", "discover_skill_md",
           "make_knowledge_skill", "ingest_knowledge_corpus", "EXTERNAL"]
