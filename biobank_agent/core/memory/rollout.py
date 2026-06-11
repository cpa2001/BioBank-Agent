"""JSONL rollout writer for event-stream persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class RolloutWriter:
    """Append JSON-serialisable runtime events to a rollout file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Any) -> None:
        payload = self._to_jsonable(event)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return cls._to_jsonable(value.to_dict())
        if is_dataclass(value):
            return cls._to_jsonable(asdict(value))
        if isinstance(value, dict):
            return {str(k): cls._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._to_jsonable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)


__all__ = ["RolloutWriter"]
