"""Best-effort notifications for long-running / paused work (issues #1, #3).

In headless / unattended runs the operator has walked away, so a finished
background job or a run that paused for input must leave a durable trace and,
optionally, ping an external channel. ``notify`` is intentionally fire-and-forget:
it never raises, so a notification failure can't break execution.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def notify(event: str, *, title: str = "", body: str = "", settings: Any = None) -> None:
    """Emit a best-effort notification through ordered sinks; never raises.

    Sinks: (1) append a JSONL record to ``<memory_dir>/notifications.jsonl``
    (always, so there is a durable trail); (2) run ``settings.notify_command`` if
    configured (the message is piped on stdin); (3) ring the terminal bell on a TTY.
    """
    if settings is not None and not bool(getattr(settings, "notify_enabled", True)):
        return
    record = {"ts": time.time(), "event": str(event), "title": str(title), "body": str(body)}

    try:
        memory_dir = getattr(settings, "memory_dir", None) if settings is not None else None
        base = Path(memory_dir) if memory_dir else Path.home() / ".biobank_agent"
        base.mkdir(parents=True, exist_ok=True)
        with (base / "notifications.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass

    command = str(getattr(settings, "notify_command", "") or "") if settings is not None else ""
    if command.strip():
        try:
            subprocess.run(command, shell=True, input=f"{title}: {body}".strip(),
                           text=True, timeout=15, capture_output=True, check=False)
        except Exception:
            pass

    try:
        if sys.stderr.isatty():
            sys.stderr.write("\a")
            sys.stderr.flush()
    except Exception:
        pass
