"""Self-evolution layer.

Reflexion 4-actions, tool_learner pattern mining, patch_classifier,
auto_merger with risk-graded LOW/MEDIUM/HIGH handling.
"""

from .pattern_mining import EvolutionPatternRun, run_pattern_mining

__all__ = ["EvolutionPatternRun", "run_pattern_mining"]
