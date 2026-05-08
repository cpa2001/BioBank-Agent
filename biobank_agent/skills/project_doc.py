"""Read-only access to curated project documentation.

This skill gives the agent a safe way to discover and read the Markdown files
that define repository usage, data coverage, plugin behavior, and agent
operating rules. It intentionally excludes raw research notes and generated
outputs.
"""

from __future__ import annotations

from pathlib import Path
import re

from biobank_agent.registry import skill


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROOT_DOCS = {"README.md", "CHANGELOG.md", "CLAUDE.md"}
_DOC_DIRS = (
    Path("docs"),
    Path("plugins/biobank-agent/skills"),
    Path("plugins/biobank-agent-claude/commands"),
    Path("plugins/biobank-agent-claude/agents"),
)
_TEXT_EXTENSIONS = {".md", ".mdx", ".txt"}
_EXCLUDED_PARTS = {"deep_research", "__pycache__"}


def _within_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(_REPO_ROOT)
        return True
    except ValueError:
        return False


def _is_allowed_doc(path: Path) -> bool:
    if not _within_repo(path):
        return False
    rel = path.resolve().relative_to(_REPO_ROOT)
    if any(part in _EXCLUDED_PARTS for part in rel.parts):
        return False
    if rel.name in _ROOT_DOCS and len(rel.parts) == 1:
        return True
    if path.suffix.lower() not in _TEXT_EXTENSIONS:
        return False
    return any(rel.is_relative_to(doc_dir) for doc_dir in _DOC_DIRS)


def _iter_project_docs() -> list[Path]:
    docs: list[Path] = []
    for filename in sorted(_ROOT_DOCS):
        path = _REPO_ROOT / filename
        if path.exists() and _is_allowed_doc(path):
            docs.append(path)

    for doc_dir in _DOC_DIRS:
        base = _REPO_ROOT / doc_dir
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and _is_allowed_doc(path):
                docs.append(path)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in docs:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _resolve_project_doc(path: str) -> Path | None:
    raw = str(path or "").strip().lstrip("/")
    if not raw:
        return None
    candidate = (_REPO_ROOT / raw).resolve()
    if candidate.exists() and candidate.is_file() and _is_allowed_doc(candidate):
        return candidate
    return None


def _title_for(path: Path, text: str | None = None) -> str:
    if text is None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return path.stem
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip() or path.stem
    return path.stem


def _snippet(text: str, query: str, max_len: int = 360) -> tuple[str, int | None]:
    if not query:
        preview = re.sub(r"\s+", " ", text).strip()
        return preview[:max_len], None

    lower = text.lower()
    query_lower = query.lower()
    idx = lower.find(query_lower)
    if idx < 0:
        terms = [term for term in re.findall(r"[a-zA-Z0-9_/-]{3,}", query_lower)]
        idx = min((lower.find(term) for term in terms if lower.find(term) >= 0), default=0)

    line_no = text.count("\n", 0, idx) + 1 if idx else 1
    start = max(0, idx - max_len // 3)
    end = min(len(text), start + max_len)
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    return snippet, line_no


def _score(text: str, query: str, path: Path) -> int:
    haystack = f"{path.as_posix()}\n{text}".lower()
    terms = [term for term in re.findall(r"[a-zA-Z0-9_/-]{2,}", query.lower())]
    if not terms:
        return 0
    score = 0
    for term in terms:
        if term in haystack:
            score += 1
        score += haystack.count(term)
    return score


@skill(
    name="project_doc",
    description=(
        "List, search, or read curated project Markdown documentation. Use this "
        "when the agent needs repository guidance, data reference notes, plugin "
        "instructions, custom skill docs, or architecture docs."
    ),
    parameters={
        "mode": {
            "type": "string",
            "description": "One of list, search, or read. Defaults from path/query.",
            "default": "",
        },
        "query": {
            "type": "string",
            "description": "Search terms for documentation lookup.",
            "default": "",
        },
        "path": {
            "type": "string",
            "description": "Repository-relative Markdown path to read.",
            "default": "",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum list/search results to return.",
            "default": 12,
        },
        "max_chars": {
            "type": "integer",
            "description": "Maximum characters returned when reading a document.",
            "default": 8000,
        },
    },
    required=[],
)
def project_doc(
    mode: str = "",
    query: str = "",
    path: str = "",
    limit: int = 12,
    max_chars: int = 8000,
    *,
    ctx=None,
) -> dict:
    """List, search, or read curated repository documentation."""
    del ctx
    mode = str(mode or "").strip().lower()
    query = str(query or "").strip()
    path = str(path or "").strip()
    limit = max(1, min(int(limit or 12), 50))
    max_chars = max(500, min(int(max_chars or 8000), 50000))

    if mode not in {"", "list", "search", "read"}:
        return {"status": "error", "error": "mode must be one of list, search, or read"}
    if not mode:
        mode = "read" if path else "search" if query else "list"

    docs = _iter_project_docs()

    if mode == "list":
        entries = []
        for doc_path in docs[:limit]:
            rel = doc_path.relative_to(_REPO_ROOT).as_posix()
            try:
                text = doc_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            entries.append({
                "path": rel,
                "title": _title_for(doc_path, text),
                "chars": len(text),
            })
        return {"status": "success", "mode": "list", "total": len(docs), "results": entries}

    if mode == "read":
        doc_path = _resolve_project_doc(path)
        if doc_path is None:
            return {
                "status": "error",
                "error": "Document is missing or outside the curated documentation set.",
                "path": path,
                "allowed_roots": ["README.md", "CHANGELOG.md", "CLAUDE.md", "docs/", "plugins/"],
            }
        text = doc_path.read_text(encoding="utf-8", errors="replace")
        rel = doc_path.relative_to(_REPO_ROOT).as_posix()
        return {
            "status": "success",
            "mode": "read",
            "path": rel,
            "title": _title_for(doc_path, text),
            "content": text[:max_chars],
            "truncated": len(text) > max_chars,
            "chars": len(text),
        }

    if not query:
        return {
            "status": "error",
            "error": "Search mode requires a query. Use mode=list to enumerate available docs.",
        }

    matches = []
    for doc_path in docs:
        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        score = _score(text, query, doc_path.relative_to(_REPO_ROOT))
        if score <= 0:
            continue
        snippet, line = _snippet(text, query)
        matches.append({
            "path": doc_path.relative_to(_REPO_ROOT).as_posix(),
            "title": _title_for(doc_path, text),
            "score": score,
            "line": line,
            "snippet": snippet,
        })

    matches.sort(key=lambda item: (-item["score"], item["path"]))
    return {
        "status": "success",
        "mode": "search",
        "query": query,
        "total": len(matches),
        "results": matches[:limit],
    }
