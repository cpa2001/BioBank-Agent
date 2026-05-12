"""Compatibility package for the Biobank Agent CLI.

Historically the CLI lived in ``biobank_agent/cli.py``. v3 work needs real
submodules such as ``biobank_agent.cli.commands`` and ``biobank_agent.cli.tui``,
so the legacy module now lives at ``biobank_agent.cli_legacy``. This package
aliases itself to that module while preserving ``__path__`` so submodules remain
importable and existing monkeypatches against ``biobank_agent.cli`` still affect
the live CLI globals.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

_legacy = import_module("biobank_agent.cli_legacy")
_legacy.__path__ = [str(Path(__file__).parent)]  # allow biobank_agent.cli.*
sys.modules[__name__] = _legacy
