"""Abstract interface for agent self-evolution.

Stub for:
- Automatic skill creation (code generation with AST safety)
- Hyperparameter memory and auto-tuning
- Pipeline macro recording and replay
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional


class EvolutionInterface(ABC):
    """Abstract base for agent self-improvement capabilities."""

    @abstractmethod
    def save_pipeline(self, name: str, steps: list[dict]) -> None:
        """Record a named analysis pipeline for replay."""

    @abstractmethod
    def replay_pipeline(self, name: str, overrides: Optional[dict] = None) -> list[dict]:
        """Replay a saved pipeline with optional parameter overrides."""

    @abstractmethod
    def suggest_improvement(self, analysis_records: list[dict]) -> str:
        """Analyse session history and suggest improvements."""

    @abstractmethod
    def create_skill(self, name: str, description: str, code: str) -> dict:
        """Create a new skill from generated code (AST-gated, user-approved)."""
