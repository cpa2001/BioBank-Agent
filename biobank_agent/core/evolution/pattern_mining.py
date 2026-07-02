"""Scheduled self-evolution pattern mining artifacts.

This module is deliberately non-mutating. It turns the in-memory
``ToolLearner`` failure history into JSON/Markdown artifacts that can be run by
cron or CI before any generated patch is considered.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EvolutionPatternRun:
    """One scheduled pattern-mining run."""

    status: str
    timestamp: str
    output_dir: str
    min_count: int
    patterns: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    artifacts: dict[str, str] = field(default_factory=dict)
    # Repeated runs of consecutive SUCCESSFUL calls — candidates for auto-captured wrapper skills
    # (self-evolution). Additive: absent from a learner that predates success-sequence mining.
    sequences: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_patterns(self) -> int:
        return len(self.patterns)

    @property
    def n_proposals(self) -> int:
        return len(self.proposals)

    @property
    def n_sequences(self) -> int:
        return len(self.sequences)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["n_patterns"] = self.n_patterns
        data["n_proposals"] = self.n_proposals
        data["n_sequences"] = self.n_sequences
        return data


def run_pattern_mining(
    learner: Any,
    *,
    output_dir: str | Path = "reports/eval/evolution",
    min_count: int = 3,
    write_artifacts: bool = True,
) -> EvolutionPatternRun:
    """Mine repeated failure patterns and optionally write audit artifacts."""
    patterns = [
        _pattern_to_dict(pattern)
        for pattern in (learner.mine_failure_patterns(min_count=min_count) or [])
    ]
    proposals = [
        _proposal_to_dict(proposal)
        for proposal in (learner.auto_propose_skill_improvement(min_count=min_count) or [])
    ]
    # Success-sequence mining is additive and duck-typed: a learner without it simply yields none, so the
    # failure-mining status semantics are unchanged (sequences are opportunities surfaced for review).
    sequence_miner = getattr(learner, "mine_success_sequences", None)
    sequences = (
        [_sequence_to_dict(seq) for seq in (sequence_miner(min_count=min_count) or [])]
        if callable(sequence_miner) else []
    )
    result = EvolutionPatternRun(
        status="NEEDS_REVIEW" if patterns else "PASS",
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        output_dir=str(output_dir),
        min_count=int(min_count),
        patterns=patterns,
        proposals=proposals,
        sequences=sequences,
    )
    if write_artifacts:
        _write_artifacts(result, Path(output_dir))
    return result


def _write_artifacts(result: EvolutionPatternRun, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_json = output_dir / f"evolution_patterns_{result.timestamp}.json"
    latest_json = output_dir / "latest.json"
    history_jsonl = output_dir / "evolution_pattern_history.jsonl"
    run_md = output_dir / f"evolution_patterns_{result.timestamp}.md"
    artifacts = {
        "run_json": str(run_json),
        "latest_json": str(latest_json),
        "history_jsonl": str(history_jsonl),
        "run_md": str(run_md),
    }
    result.artifacts.update(artifacts)
    payload = result.to_dict()

    run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    run_md.write_text(_render_markdown(result), encoding="utf-8")
    history = {
        "timestamp": result.timestamp,
        "status": result.status,
        "n_patterns": result.n_patterns,
        "n_proposals": result.n_proposals,
        "min_count": result.min_count,
        "run_json": str(run_json),
        "run_md": str(run_md),
    }
    with history_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(history, sort_keys=True) + "\n")
    return artifacts


def _pattern_to_dict(pattern: Any) -> dict[str, Any]:
    return {
        "skill_name": str(getattr(pattern, "skill_name", "")),
        "error_signature": str(getattr(pattern, "error_signature", "")),
        "count": int(getattr(pattern, "count", 0) or 0),
        "suggested_action": str(getattr(pattern, "suggested_action", "")),
        "examples": _jsonable(getattr(pattern, "examples", []) or []),
    }


def _sequence_to_dict(sequence: Any) -> dict[str, Any]:
    return {
        "sequence": [str(s) for s in getattr(sequence, "sequence", ()) or ()],
        "count": int(getattr(sequence, "count", 0) or 0),
        "length": int(getattr(sequence, "length", 0) or 0),
        "examples": _jsonable(getattr(sequence, "examples", []) or []),
    }


def _proposal_to_dict(proposal: Any) -> dict[str, Any]:
    if not isinstance(proposal, dict):
        return {"raw": str(proposal)}
    keep = {
        "skill",
        "failure_count",
        "error_signature",
        "suggested_action",
        "risk",
        "risk_reason",
        "target_path",
        "candidate_patch_required",
        "apply_mode",
    }
    data = {key: _jsonable(value) for key, value in proposal.items() if key in keep}
    patch = str(proposal.get("candidate_patch", ""))
    if patch:
        data["candidate_patch_preview"] = patch[:2000]
    return data


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return str(value)


def _render_markdown(result: EvolutionPatternRun) -> str:
    lines = [
        "# Biobank Agent Evolution Pattern Mining",
        "",
        f"- Status: {result.status}",
        f"- Timestamp: {result.timestamp}",
        f"- Minimum repeated failures: {result.min_count}",
        f"- Patterns: {result.n_patterns}",
        f"- Proposals: {result.n_proposals}",
        "",
        "## Repeated Failure Patterns",
        "",
    ]
    if not result.patterns:
        lines.append("No repeated failure pattern reached the configured threshold.")
    for idx, pattern in enumerate(result.patterns, start=1):
        lines.extend([
            f"### {idx}. {pattern.get('skill_name', 'unknown')}",
            "",
            f"- Count: {pattern.get('count', 0)}",
            f"- Error signature: `{pattern.get('error_signature', '')}`",
            f"- Suggested action: {pattern.get('suggested_action', '')}",
            "",
        ])
    if result.proposals:
        lines.extend(["", "## Review-Only Proposals", ""])
        for idx, proposal in enumerate(result.proposals, start=1):
            lines.extend([
                f"### {idx}. {proposal.get('skill', 'unknown')}",
                "",
                f"- Risk: {proposal.get('risk', 'unknown')}",
                f"- Target: `{proposal.get('target_path', '')}`",
                f"- Suggested action: {proposal.get('suggested_action', '')}",
                "",
            ])
    if result.sequences:
        lines.extend(["", "## Frequent Successful Sequences (auto-capture candidates)", ""])
        for idx, seq in enumerate(result.sequences, start=1):
            steps = " → ".join(seq.get("sequence", []))
            lines.append(f"{idx}. `{steps}` (observed {seq.get('count', 0)}x)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["EvolutionPatternRun", "run_pattern_mining"]
