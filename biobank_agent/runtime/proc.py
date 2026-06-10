"""Streaming subprocess runner — the shared execution substrate.

Replaces the post-hoc ``capture_output=True`` pattern with line-by-line streaming
so the agent (and the user) can see a long bioinformatics command's progress
live, while always persisting the full output to a log file and keeping a bounded
tail for the LLM-facing result. Runs each command in its own process group so a
timeout/cancel reaps the whole child tree (gatk/plink spawn children).

Both the foreground tools and the background ``JobManager`` build on this.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# (stream_name, line) -> None ; stream in {"stdout", "stderr"}; line has no trailing newline.
LineSink = Callable[[str, str], None]


@dataclass
class ProcResult:
    cmd: list[str] | str
    returncode: int | None
    stdout_tail: str = ""
    stderr_tail: str = ""
    ok: bool = False
    timed_out: bool = False
    log_path: str | None = None
    pid: int | None = None
    duration_s: float = 0.0


class _Tail:
    """Bounded last-N-chars buffer assembled from streamed lines (O(1) amortized)."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max(0, int(max_chars))
        self._lines: deque[str] = deque()
        self._size = 0

    def add(self, line: str) -> None:
        self._lines.append(line)
        self._size += len(line)
        while self._size > self.max_chars and len(self._lines) > 1:
            self._size -= len(self._lines.popleft())

    def text(self) -> str:
        return "".join(self._lines)[-self.max_chars:] if self.max_chars else ""


def _reader(stream_name: str, pipe, tail: _Tail, sink: Optional[LineSink],
            log_handle, log_lock: threading.Lock) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            tail.add(line)
            if log_handle is not None:
                with log_lock:
                    log_handle.write(line if line.endswith("\n") else line + "\n")
            if sink is not None:
                try:
                    sink(stream_name, line.rstrip("\n"))
                except Exception:
                    pass  # a display sink must never break execution
    except (ValueError, OSError):
        pass  # pipe closed under us during kill
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _kill_group(proc: subprocess.Popen, *, grace_s: float = 5.0) -> None:
    """SIGTERM the process group, then SIGKILL after a grace period."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        return
    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


def run_streaming(
    cmd: list[str] | str,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | float | None = None,
    shell: bool = False,
    line_sink: Optional[LineSink] = None,
    log_path: str | Path | None = None,
    tail_chars: int = 5000,
    cancel_event: Optional[threading.Event] = None,
) -> ProcResult:
    """Run ``cmd`` with live line streaming, a persisted log, and a bounded tail.

    ``timeout=None`` means no wall-clock cap (background jobs). On timeout or when
    ``cancel_event`` is set, the whole process group is terminated. ``line_sink``
    receives every (stream, line) live; ``log_path`` (if given) gets the full
    interleaved output. Returns a :class:`ProcResult`.
    """
    start = time.monotonic()
    log_handle = None
    log_str = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", errors="replace")
        log_str = str(log_path)

    out_tail = _Tail(tail_chars)
    err_tail = _Tail(tail_chars)
    log_lock = threading.Lock()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            shell=shell,
            start_new_session=True,  # own process group -> killpg reaps children
        )
    except (OSError, ValueError) as exc:
        if log_handle is not None:
            log_handle.close()
        return ProcResult(cmd=cmd, returncode=None, stdout_tail="",
                          stderr_tail=str(exc)[-tail_chars:], ok=False,
                          log_path=log_str, duration_s=time.monotonic() - start)

    t_out = threading.Thread(target=_reader, args=("stdout", proc.stdout, out_tail, line_sink, log_handle, log_lock), daemon=True)
    t_err = threading.Thread(target=_reader, args=("stderr", proc.stderr, err_tail, line_sink, log_handle, log_lock), daemon=True)
    t_out.start()
    t_err.start()

    timed_out = False
    try:
        while True:
            if proc.poll() is not None:
                break
            elapsed = time.monotonic() - start
            if timeout is not None and elapsed >= timeout:
                timed_out = True
                _kill_group(proc)
                break
            if cancel_event is not None and cancel_event.is_set():
                _kill_group(proc)
                break
            time.sleep(0.05)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _kill_group(proc, grace_s=0.0)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    finally:
        t_out.join(timeout=5)
        t_err.join(timeout=5)
        if log_handle is not None:
            with log_lock:
                try:
                    log_handle.flush()
                finally:
                    log_handle.close()

    rc = proc.returncode
    stderr_text = err_tail.text()
    if timed_out:
        stderr_text = (f"Timed out after {timeout}s. " + stderr_text)[-tail_chars:]
    return ProcResult(
        cmd=cmd,
        returncode=rc,
        stdout_tail=out_tail.text(),
        stderr_tail=stderr_text,
        ok=(rc == 0 and not timed_out),
        timed_out=timed_out,
        log_path=log_str,
        pid=proc.pid,
        duration_s=time.monotonic() - start,
    )
