"""A/B harness: keep the adversarial council only if it beats the symmetric one.

Runs a fixed objective set through two planning modes, scores each result with an injected
scorer, and tallies wins. This is the empirical gate for the alternative planning pipeline — it
stays behind its flag until it wins here. Pure over injected run/score functions, so it runs
offline in tests and against real models alike (e.g. scoring with runtime/run_eval.py).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


def ab_compare(
    items: Iterable[Any],
    run_a: Callable[[Any], Any],
    run_b: Callable[[Any], Any],
    score_fn: Callable[[Any], float],
    *,
    label_a: str = "a",
    label_b: str = "b",
) -> dict[str, Any]:
    """Run each item through both modes, score both, and tally wins (pure, deterministic).

    Returns per-mode win counts, mean scores, and the overall winner (or ``"tie"``)."""
    results: list[dict[str, Any]] = []
    wins_a = wins_b = ties = 0
    sum_a = sum_b = 0.0
    for item in items:
        score_a = float(score_fn(run_a(item)))
        score_b = float(score_fn(run_b(item)))
        sum_a += score_a
        sum_b += score_b
        if score_a > score_b:
            wins_a += 1
        elif score_b > score_a:
            wins_b += 1
        else:
            ties += 1
        results.append({"item": item, label_a: score_a, label_b: score_b})

    n = len(results) or 1
    winner = label_a if wins_a > wins_b else label_b if wins_b > wins_a else "tie"
    return {
        "wins": {label_a: wins_a, label_b: wins_b, "tie": ties},
        "mean": {label_a: round(sum_a / n, 4), label_b: round(sum_b / n, 4)},
        "winner": winner,
        "n": len(results),
        "results": results,
    }


__all__ = ["ab_compare"]
