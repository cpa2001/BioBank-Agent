"""Platform detection — CPU/GPU, arch, memory info.

Used by CLI /status and skills that need GPU-awareness.
"""

from __future__ import annotations

import platform
import sys
from typing import Any


def has_gpu() -> bool:
    """Check if a CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return True
    except (ImportError, Exception):
        pass
    return False


def gpu_info() -> dict[str, Any]:
    """Get GPU details if available."""
    if not has_gpu():
        return {"available": False}
    try:
        import torch
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1),
        }
    except Exception:
        return {"available": True, "device_count": 1, "detail": "unknown"}


def platform_info() -> dict[str, Any]:
    """Get platform information."""
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "arch": platform.machine(),
        "processor": platform.processor() or "unknown",
        "system": platform.system(),
    }

    # Memory info (platform-specific)
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["memory_total_gb"] = round(mem.total / 1e9, 1)
        info["memory_available_gb"] = round(mem.available / 1e9, 1)
    except ImportError:
        pass

    # GPU
    gpu = gpu_info()
    info["gpu"] = gpu

    return info


def platform_summary() -> str:
    """One-line platform summary for display."""
    info = platform_info()
    parts = [
        f"Python {info['python_version']}",
        info["arch"],
        info["system"],
    ]
    if info.get("memory_total_gb"):
        parts.append(f"{info['memory_total_gb']}GB RAM")
    if info["gpu"]["available"]:
        parts.append(f"GPU: {info['gpu'].get('device_name', 'yes')}")
    else:
        parts.append("No GPU")
    return " | ".join(parts)
