"""Live entry point for external GitHub skill ingestion.

Demand-driven and default-off: this skill clones a pinned GitHub corpus via the gated
``shell_exec`` seam (never in-process git), then ingests its ``SKILL.md`` files as deferred
KNOWLEDGE skills (no third-party code is written or executed — see runtime/skill_ingest.py).
Activation requires the ``external_skill_ingestion_enabled`` flag AND an immutable commit SHA.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

from biobank_agent.registry import skill

_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
# A git URL with no shell metacharacters — repo_url is interpolated into a shell=True command,
# so it must be validated AND quoted (defense in depth) to prevent command injection.
_REPO_URL_RE = re.compile(r"^(https://|git://|ssh://|git@)[A-Za-z0-9._:/~@-]+$")


def _repo_slug(repo_url: str) -> str:
    tail = re.sub(r"\.git$", "", str(repo_url).rstrip("/")).rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9_]+", "_", tail.lower()).strip("_") or "corpus"


@skill(
    name="ingest_github_skills",
    description=(
        "Ingest a community skill library from a pinned GitHub commit as deferred, "
        "trust='external' KNOWLEDGE skills (SKILL.md metadata only — no third-party code "
        "runs). Off unless external_skill_ingestion_enabled is set; requires a commit SHA."
    ),
    parameters={
        "repo_url": {"type": "string", "description": "Git URL of the skill corpus"},
        "ref": {"type": "string", "description": "Immutable commit SHA to pin (required)", "default": ""},
        "license": {"type": "string", "description": "Corpus license identifier", "default": ""},
    },
    required=["repo_url"],
)
def ingest_github_skills(repo_url: str, ref: str = "", license: str = "", *, ctx=None) -> dict:
    from biobank_agent.skills.local_exec import shell_exec
    from biobank_agent.runtime.skill_ingest import (
        SkillPackage, discover_skill_md, ingest_knowledge_corpus,
    )

    settings = getattr(ctx, "settings", None)
    if not getattr(settings, "external_skill_ingestion_enabled", False):
        return {"status": "disabled",
                "message": "Set external_skill_ingestion_enabled to enable corpus ingestion."}

    repo_url = str(repo_url or "").strip()
    if not _REPO_URL_RE.match(repo_url):
        return {"status": "invalid_repo_url",
                "message": "repo_url must be a plain https/git/ssh URL with no shell metacharacters."}

    ref = str(ref or "").strip()
    if not _SHA_RE.match(ref):
        return {"status": "needs_commit_pin",
                "message": "Pin an immutable commit SHA (ref) — floating branches are not ingestable."}

    base = Path(getattr(settings, "external_skills_dir", Path("./external_skills"))).expanduser()
    corpus = _repo_slug(repo_url)
    dest = base / corpus
    base.mkdir(parents=True, exist_ok=True)

    if not (dest / ".git").exists():
        clone = shell_exec(command=f"git clone {shlex.quote(repo_url)} {shlex.quote(str(dest))}",
                           cwd=str(base), write_policy="environment_write", confirmed=True, ctx=ctx)
        if clone.get("status") != "success":
            return {"status": "clone_failed", "error": clone.get("stderr", "")[:500]}
    checkout = shell_exec(command=f"git checkout {shlex.quote(ref)}", cwd=str(dest),
                          write_policy="environment_write", confirmed=True, ctx=ctx)
    if checkout.get("status") != "success":
        return {"status": "checkout_failed", "ref": ref, "error": checkout.get("stderr", "")[:500]}

    sources = discover_skill_md(dest)
    package = SkillPackage(name=corpus, source_url=str(repo_url), commit=ref, license=license)
    result = ingest_knowledge_corpus(sources, package)

    sidecar = dest / "SKILL_PACKAGE.json"
    sidecar.write_text(json.dumps(result["package"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["sidecar"] = str(sidecar)
    return result
