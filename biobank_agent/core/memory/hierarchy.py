"""Minimal hierarchical memory collector.

The v3 plan separates memory into global, project, and just-in-time
blocks. The legacy ``LongTermMemory`` remains the backing store; this
module gives new runtime code a stable structured API without forcing a
large migration in one patch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MemoryBlock:
    scope: str
    content: str
    path: str = ""


class MemoryInjector:
    """Collect global and project memory blocks for prompt injection."""

    def __init__(self, *, global_paths: Iterable[Path] | None = None) -> None:
        self.global_paths = [Path(p) for p in (global_paths or [])]

    def collect(self, cwd: str | Path) -> list[MemoryBlock]:
        cwd = Path(cwd)
        blocks: list[MemoryBlock] = []
        for path in self.global_paths:
            self._append_if_readable(blocks, "global", path)
        for name in ("BIOBANK_AGENT.md", "AGENTS.md", "README.md"):
            self._append_if_readable(blocks, "project", cwd / name)
        return blocks

    @staticmethod
    def _append_if_readable(blocks: list[MemoryBlock], scope: str, path: Path) -> None:
        try:
            if path.exists() and path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
            else:
                return
        except OSError:
            return
        if text:
            blocks.append(MemoryBlock(scope=scope, content=text[:8000], path=str(path)))


__all__ = ["MemoryBlock", "MemoryInjector"]
