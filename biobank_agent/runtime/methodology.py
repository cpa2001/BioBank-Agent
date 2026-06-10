"""Methodological-soundness reviewer.

Scientific correctness is not exit-code 0: a GWAS/PheWAS that reports raw p-values
with no multiple-testing correction, an association with n below any power, or an
effect estimate with no confidence interval is *wrong* even when it "ran". This
pure, dependency-free reviewer scans an analysis result (and/or its narrative) for
consensus statistical sins and returns flags the verification mesh can hard-block
on (``severity='block'``) or surface as advisories (``severity='advisory'``).
"""

from __future__ import annotations

from typing import Any

_CORRECTION_HINTS = (
    "fdr", "bonferroni", "benjamini", "hochberg", "q_value", "qvalue", "q_val",
    "p_adj", "padj", "p_adjusted", "adjusted_p", "p_corrected", "p_bonferroni",
    "p_fdr", "holm", "sidak",
)
_CI_HINTS = ("ci", "conf_int", "confidence_interval", "ci_low", "ci_lower", "ci95",
             "ci_95", "lower_ci", "l95", "u95", "ci_upper", "hr_ci", "or_ci")
_EFFECT_HINTS = ("odds_ratio", "beta", "hazard_ratio", "coef", "effect_size", "log_or")
_MIN_GROUP_N = 10
_MIN_TOTAL_N = 20


def _walk_numbers(payload: Any, key_substr: str, limit: int = 100000) -> list[float]:
    out: list[float] = []

    def _rec(node: Any, depth: int = 0) -> None:
        if len(out) >= limit or depth > 6:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and key_substr in str(k).lower():
                    out.append(float(v))
                else:
                    _rec(v, depth + 1)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _rec(v, depth + 1)

    _rec(payload)
    return out


def _has_key_hint(payload: Any, hints: tuple[str, ...], depth: int = 0) -> bool:
    if depth > 6:
        return False
    if isinstance(payload, dict):
        for k, v in payload.items():
            kl = str(k).lower()
            if any(h in kl for h in hints):
                return True
            if _has_key_hint(v, hints, depth + 1):
                return True
    elif isinstance(payload, (list, tuple)):
        return any(_has_key_hint(v, hints, depth + 1) for v in payload)
    return False


def review_methodology(payload: Any, *, text: str = "", n_test_threshold: int = 20) -> list[dict[str, Any]]:
    """Return methodology flags: ``[{issue, severity ('block'|'advisory'), detail}]``.
    Heuristic and conservative — only clear consensus errors are 'block'."""
    flags: list[dict[str, Any]] = []
    lower = (text or "").lower()

    # 1) Uncorrected multiple testing.
    p_like = [p for p in _walk_numbers(payload, "p") if 0.0 <= p <= 1.0]
    n_tests = len(p_like)
    n_tests_field = _walk_numbers(payload, "n_test")
    if n_tests_field:
        n_tests = max(n_tests, int(max(n_tests_field)))
    corrected = _has_key_hint(payload, _CORRECTION_HINTS) or any(
        h in lower for h in ("fdr", "bonferroni", "benjamini", "multiple testing", "multiple-testing", "q-value", "adjusted p")
    )
    if n_tests >= n_test_threshold and not corrected:
        flags.append({
            "issue": "uncorrected_multiple_testing", "severity": "block",
            "detail": f"{n_tests} tests reported with no multiple-testing correction. Apply BH-FDR or "
                      "Bonferroni before calling hits significant.",
        })

    # 2) Sample size below power.
    # n == 0 means "not reported", not "zero samples"; only a positive-but-small count
    # is genuinely underpowered.
    for key, label in (("n_cases", "cases"), ("n_controls", "controls"), ("n_case", "cases"), ("n_control", "controls")):
        small = [int(v) for v in _walk_numbers(payload, key) if 0 < v < _MIN_GROUP_N]
        if small:
            flags.append({
                "issue": "underpowered_group", "severity": "block" if min(small) < 5 else "advisory",
                "detail": f"{label} group n={min(small)} below a usable power floor (<{_MIN_GROUP_N}); "
                          "results from this arm are exploratory only.",
            })
            break
    totals = [int(t) for t in (_walk_numbers(payload, "n_samples") + _walk_numbers(payload, "n_total")) if t > 0]
    if totals and min(totals) < _MIN_TOTAL_N:
        flags.append({"issue": "underpowered_total", "severity": "advisory",
                      "detail": f"total n={min(totals)} (<{_MIN_TOTAL_N}) — interpret with caution."})

    # 3) Effect estimates without confidence intervals.
    has_effect = _has_key_hint(payload, _EFFECT_HINTS) or any(h in lower for h in ("odds ratio", "hazard ratio", "beta ="))
    has_ci = _has_key_hint(payload, _CI_HINTS) or any(h in lower for h in ("95% ci", "confidence interval", "ci:"))
    if has_effect and not has_ci:
        flags.append({"issue": "missing_confidence_interval", "severity": "advisory",
                      "detail": "Effect estimates reported without confidence intervals — add 95% CIs."})

    # 4) Genomic inflation without population-structure correction.
    lam = _walk_numbers(payload, "lambda_gc")
    if lam and max(lam) > 1.1:
        pc_corrected = _has_key_hint(payload, ("pc1", "pc_", "covar", "principal_component")) or any(
            h in lower for h in ("principal component", "pc covariate", "pca covariate", "mixed model", "mixed-model")
        )
        if not pc_corrected:
            flags.append({"issue": "unaddressed_stratification", "severity": "advisory",
                          "detail": f"lambda_gc={max(lam):.2f}>1.1 with no PC/mixed-model correction in evidence — "
                                    "add principal-component covariates (vcf_pca) or a mixed model."})
    return flags


def methodology_blocks(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in (flags or []) if f.get("severity") == "block"]
