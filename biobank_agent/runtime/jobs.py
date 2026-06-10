"""Background job manager for long-running commands (issue #1).

A long bioinformatics run (plink/gatk/WGS pipeline) should not block the agent or
die when the CLI is interrupted. ``JobManager`` launches such a command as a
DETACHED process in its own session, redirecting output straight to a per-job log
file (no pipe reader thread to die on CLI exit), persists the pid + state under
``<workspace>/.biobank_jobs/<job_id>/``, and lets the agent poll / tail / wait /
cancel it. On startup ``reattach()`` reconciles jobs left running by a prior run.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

JOB_STATES = {"running", "done", "failed", "timeout", "cancelled", "ended"}
_TERMINAL_STATES = {"done", "failed", "timeout", "cancelled", "ended"}


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, OSError):
        return False
    return True


def _kill_pid_group(pid: int | None, *, force: bool = False) -> None:
    if not pid:
        return
    try:
        pgid = os.getpgid(int(pid))
    except (ProcessLookupError, OSError):
        return
    try:
        os.killpg(pgid, signal.SIGKILL if force else signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass


@dataclass
class JobRecord:
    job_id: str
    cmd: list[str] | str
    cwd: str
    state: str = "running"
    pid: int | None = None
    returncode: int | None = None
    started_at: float = 0.0
    ended_at: float | None = None
    log_path: str = ""
    tool: str = ""
    label: str = ""
    timeout: int | None = None
    tail_stdout: str = ""
    tail_stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in dict(data).items() if k in known})

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES


class JobManager:
    """Owns detached background jobs under a workspace jobs directory."""

    def __init__(self, jobs_root: str | Path, *, bus: Any = None, notifier: Any = None,
                 clock: Any = time.time) -> None:
        self.jobs_root = Path(jobs_root)
        self.bus = bus
        self.notifier = notifier  # optional callable(event, title, body)
        self._clock = clock
        self._lock = threading.Lock()
        self._counter = 0
        # job_ids whose waiter thread is alive in THIS process; _reconcile must
        # defer to the waiter for those (it owns the terminal state) and only
        # reclaim jobs with no live waiter (e.g. orphaned by a CLI restart).
        self._active: set[str] = set()

    # ---------------------------------------------------------------- internals
    def _new_job_id(self) -> str:
        with self._lock:
            self._counter += 1
            n = self._counter
        return f"{int(self._clock())}_{n:03d}"

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def _state_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "state.json"

    def _write_state(self, rec: JobRecord) -> None:
        path = self._state_path(rec.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec.to_dict(), ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(path)  # atomic-ish

    def _read_state(self, job_id: str) -> JobRecord | None:
        path = self._state_path(job_id)
        if not path.is_file():
            return None
        try:
            return JobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, OSError, TypeError):
            return None

    def _notify(self, rec: JobRecord) -> None:
        if self.notifier is None:
            return
        event = "job_done" if rec.state == "done" else "job_failed"
        try:
            self.notifier(event, title=f"job {rec.job_id} {rec.state}",
                          body=f"{rec.label or rec.tool or 'command'} -> {rec.state} (rc={rec.returncode})")
        except Exception:
            pass

    # ------------------------------------------------------------------- public
    def submit(self, cmd: list[str] | str, *, cwd: str | Path, env: dict[str, str] | None = None,
               timeout: int | None = None, tool: str = "", label: str = "", shell: bool = False) -> JobRecord:
        """Launch ``cmd`` as a detached background job and return its record.

        Output is redirected straight to the job's ``exec.log`` (no pipe reader),
        and the process gets its own session so it survives a CLI interrupt."""
        job_id = self._new_job_id()
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "exec.log"
        try:
            (job_dir / "cmd.txt").write_text(
                cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), encoding="utf-8")
        except OSError:
            pass

        rec = JobRecord(job_id=job_id, cmd=cmd, cwd=str(cwd), state="running",
                        started_at=self._clock(), log_path=str(log_path), tool=tool,
                        label=label, timeout=timeout)
        try:
            log_handle = log_path.open("w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                shell=shell,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            rec.state = "failed"
            rec.ended_at = self._clock()
            rec.tail_stderr = str(exc)[-2000:]
            self._write_state(rec)
            return rec
        rec.pid = proc.pid
        self._write_state(rec)
        self._active.add(job_id)

        def _waiter() -> None:
            timed_out = False
            try:
                if timeout:
                    try:
                        proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        _kill_pid_group(proc.pid)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            _kill_pid_group(proc.pid, force=True)
                else:
                    proc.wait()
            finally:
                try:
                    log_handle.close()
                except Exception:
                    pass
            rec.returncode = proc.returncode
            rec.ended_at = self._clock()
            rec.tail_stdout = _tail_file(log_path)
            if timed_out:
                rec.state = "timeout"
            elif rec.returncode == 0:
                rec.state = "done"
            else:
                rec.state = "failed"
            self._write_state(rec)
            self._active.discard(job_id)
            self._notify(rec)

        threading.Thread(target=_waiter, name=f"job-{job_id}", daemon=True).start()
        return rec

    def list(self) -> list[JobRecord]:
        if not self.jobs_root.is_dir():
            return []
        out: list[JobRecord] = []
        for child in sorted(self.jobs_root.iterdir()):
            if child.name == "inline":
                continue
            rec = self._read_state(child.name)
            if rec is not None:
                out.append(self._reconcile(rec))
        return out

    def status(self, job_id: str) -> JobRecord | None:
        rec = self._read_state(job_id)
        return self._reconcile(rec) if rec is not None else None

    def _reconcile(self, rec: JobRecord) -> JobRecord:
        """If a job is marked running but its pid is gone AND no waiter in this
        process owns it (e.g. orphaned by a CLI restart), record it as 'ended'
        (finished, return code unknown). A live waiter sets the real terminal
        state, so defer to it."""
        if (rec.state == "running" and rec.job_id not in self._active
                and rec.pid is not None and not _pid_alive(rec.pid)):
            rec.state = "ended"
            rec.ended_at = rec.ended_at or self._clock()
            rec.tail_stdout = rec.tail_stdout or _tail_file(Path(rec.log_path)) if rec.log_path else rec.tail_stdout
            self._write_state(rec)
        return rec

    def tail(self, job_id: str, *, n_lines: int = 40) -> str:
        rec = self._read_state(job_id)
        if rec is None or not rec.log_path:
            return ""
        return _tail_file(Path(rec.log_path), n_lines=n_lines)

    def stream(self, job_id: str, *, poll_s: float = 0.3) -> Iterator[str]:
        """Follow a job's log until it reaches a terminal state (like tail -f)."""
        rec = self._read_state(job_id)
        if rec is None or not rec.log_path:
            return
        path = Path(rec.log_path)
        pos = 0
        while True:
            if path.is_file():
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
                if chunk:
                    yield chunk
            current = self.status(job_id)
            if current is None or current.is_terminal:
                # final flush
                if path.is_file():
                    with path.open("r", encoding="utf-8", errors="replace") as fh:
                        fh.seek(pos)
                        tail = fh.read()
                    if tail:
                        yield tail
                return
            time.sleep(poll_s)

    def wait(self, job_id: str, *, timeout: float | None = None, poll_s: float = 0.2) -> JobRecord | None:
        deadline = None if timeout is None else (time.monotonic() + timeout)
        while True:
            rec = self.status(job_id)
            if rec is None or rec.is_terminal:
                return rec
            if deadline is not None and time.monotonic() >= deadline:
                return rec
            time.sleep(poll_s)

    def cancel(self, job_id: str) -> JobRecord | None:
        rec = self._read_state(job_id)
        if rec is None:
            return None
        if rec.state == "running" and rec.pid:
            _kill_pid_group(rec.pid)
            time.sleep(0.3)
            if _pid_alive(rec.pid):
                _kill_pid_group(rec.pid, force=True)
            rec.state = "cancelled"
            rec.ended_at = self._clock()
            self._write_state(rec)
        return rec

    def reattach(self) -> list[JobRecord]:
        """Reconcile jobs left by a prior process: mark dead-pid 'running' jobs as
        failed so nothing looks alive when it isn't."""
        recovered: list[JobRecord] = []
        for rec in self.list():  # list() already reconciles dead pids -> 'ended'
            if rec.state == "running" and not _pid_alive(rec.pid):
                rec.state = "failed"
                rec.ended_at = self._clock()
                self._write_state(rec)
            recovered.append(rec)
        return recovered


def _tail_file(path: Path, *, n_lines: int = 40, max_chars: int = 4000) -> str:
    try:
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    out = "\n".join(lines[-n_lines:])
    return out[-max_chars:]
