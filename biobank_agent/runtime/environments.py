"""Execution-environment backends.

A biobank scientist does not run a 6-hour GWAS on the login node — they submit it
to a cluster scheduler (Slurm/SGE/LSF) or a cloud platform (DNAnexus ``dx run``).
This module abstracts WHERE a long command runs so the agent can target the real
compute. ``LocalBackend`` runs in-process (streaming, via runtime.proc/jobs);
scheduler backends translate a command into the cluster's submit/status/cancel
syntax and parse the returned job id.

The command-construction and id-parsing are pure + unit-testable without a real
cluster; actual submission is exercised only where a scheduler is installed.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JobResources:
    cpus: int = 1
    mem_mb: int = 4096
    time_min: int = 240
    partition: str = ""
    extra: list[str] = field(default_factory=list)


def _minutes_to_hms(minutes: int) -> str:
    minutes = max(1, int(minutes))
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


class EnvironmentBackend:
    """Abstract submit/status/cancel for one execution environment."""

    name = "base"
    is_scheduler = False

    def available(self) -> bool:
        raise NotImplementedError

    def submit_argv(self, script_path: str, *, job_name: str, resources: JobResources) -> list[str]:
        raise NotImplementedError

    def parse_job_id(self, submit_stdout: str) -> str:
        raise NotImplementedError

    def status_argv(self, job_id: str) -> list[str]:
        raise NotImplementedError

    def cancel_argv(self, job_id: str) -> list[str]:
        raise NotImplementedError

    def build_script(self, command: str, *, cwd: str, job_name: str, resources: JobResources) -> str:
        """A portable bash job script (scheduler directives are passed on the CLI)."""
        return "#!/bin/bash\nset -euo pipefail\n" + f"cd {cwd}\n{command}\n"


class LocalBackend(EnvironmentBackend):
    name = "local"
    is_scheduler = False

    def available(self) -> bool:
        return True


class SlurmBackend(EnvironmentBackend):
    name = "slurm"
    is_scheduler = True

    def available(self) -> bool:
        return shutil.which("sbatch") is not None

    def submit_argv(self, script_path: str, *, job_name: str, resources: JobResources) -> list[str]:
        argv = ["sbatch", "--parsable", "--job-name", job_name,
                "--cpus-per-task", str(resources.cpus),
                "--mem", f"{resources.mem_mb}M",
                "--time", _minutes_to_hms(resources.time_min)]
        if resources.partition:
            argv += ["--partition", resources.partition]
        argv += list(resources.extra)
        argv.append(script_path)
        return argv

    def parse_job_id(self, submit_stdout: str) -> str:
        lines = (submit_stdout or "").strip().splitlines()
        if not lines:
            return ""
        return lines[0].split(";")[0].strip()

    def status_argv(self, job_id: str) -> list[str]:
        return ["squeue", "--job", str(job_id), "--noheader", "--format", "%T"]

    def cancel_argv(self, job_id: str) -> list[str]:
        return ["scancel", str(job_id)]


class SGEBackend(EnvironmentBackend):
    name = "sge"
    is_scheduler = True

    def available(self) -> bool:
        return shutil.which("qsub") is not None

    def submit_argv(self, script_path: str, *, job_name: str, resources: JobResources) -> list[str]:
        argv = ["qsub", "-terse", "-N", job_name, "-pe", "smp", str(resources.cpus),
                "-l", f"h_vmem={resources.mem_mb}M", "-l", f"h_rt={_minutes_to_hms(resources.time_min)}"]
        if resources.partition:
            argv += ["-q", resources.partition]
        argv += list(resources.extra)
        argv.append(script_path)
        return argv

    def parse_job_id(self, submit_stdout: str) -> str:
        lines = [ln for ln in (submit_stdout or "").strip().splitlines() if ln.strip()]
        if not lines:
            return ""
        m = re.search(r"\d+", lines[0])
        return m.group(0) if m else lines[0].strip()

    def status_argv(self, job_id: str) -> list[str]:
        return ["qstat", "-j", str(job_id)]

    def cancel_argv(self, job_id: str) -> list[str]:
        return ["qdel", str(job_id)]


class LSFBackend(EnvironmentBackend):
    name = "lsf"
    is_scheduler = True

    def available(self) -> bool:
        return shutil.which("bsub") is not None

    def submit_argv(self, script_path: str, *, job_name: str, resources: JobResources) -> list[str]:
        argv = ["bsub", "-J", job_name, "-n", str(resources.cpus),
                "-M", str(resources.mem_mb), "-W", str(resources.time_min)]
        if resources.partition:
            argv += ["-q", resources.partition]
        argv += list(resources.extra)
        argv += ["-i", script_path]
        return argv

    def parse_job_id(self, submit_stdout: str) -> str:
        m = re.search(r"Job <(\d+)>", submit_stdout or "")
        return m.group(1) if m else ""

    def status_argv(self, job_id: str) -> list[str]:
        return ["bjobs", "-o", "stat", "-noheader", str(job_id)]

    def cancel_argv(self, job_id: str) -> list[str]:
        return ["bkill", str(job_id)]


class DxBackend(EnvironmentBackend):
    name = "dxrun"
    is_scheduler = True

    def available(self) -> bool:
        return shutil.which("dx") is not None

    def submit_argv(self, script_path: str, *, job_name: str, resources: JobResources) -> list[str]:
        return ["dx", "run", "--name", job_name, "--brief", "-y",
                "app-cloud_workstation", "-iscript_path=" + script_path] + list(resources.extra)

    def parse_job_id(self, submit_stdout: str) -> str:
        m = re.search(r"job-[A-Za-z0-9]+", submit_stdout or "")
        if m:
            return m.group(0)
        lines = (submit_stdout or "").strip().splitlines()
        return lines[0].strip() if lines else ""

    def status_argv(self, job_id: str) -> list[str]:
        return ["dx", "describe", "--json", str(job_id)]

    def cancel_argv(self, job_id: str) -> list[str]:
        return ["dx", "terminate", str(job_id)]


_BACKENDS: dict[str, type[EnvironmentBackend]] = {
    "local": LocalBackend, "slurm": SlurmBackend, "sge": SGEBackend,
    "lsf": LSFBackend, "dxrun": DxBackend,
}


def get_backend(name: str) -> EnvironmentBackend:
    cls = _BACKENDS.get(str(name or "local").lower(), LocalBackend)
    return cls()


def detect_backend(settings: Any = None) -> EnvironmentBackend:
    """Resolve the execution backend: explicit config > first available scheduler > local."""
    configured = str(getattr(settings, "exec_backend", "auto") or "auto").lower() if settings is not None else "auto"
    if configured != "auto" and configured in _BACKENDS:
        return get_backend(configured)
    if configured == "auto":
        for name in ("slurm", "sge", "lsf", "dxrun"):
            backend = get_backend(name)
            if backend.available():
                return backend
    return LocalBackend()
