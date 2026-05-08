"""Progressive disclosure — adaptive result presentation in layers.

When the agent completes an analysis (PheWAS, model training, GWAS scan),
results are formatted in disclosure layers rather than dumped all at once:

  Layer 1 (Headline): Key finding in 1 sentence + 1 key number
  Layer 2 (Summary):  Top 5-10 results in a table
  Layer 3 (Full):     Complete results (on demand)
  Layer 4 (Methods):  Methodology, parameters, caveats (on demand)

This implements the "cognitive load control" architecture pattern where
information is revealed progressively based on user need, preventing
context overflow and attention dilution.

Usage
-----
    from biobank_agent.disclosure import ProgressivePresenter

    presenter = ProgressivePresenter()
    layers = presenter.format_phewas(results)
    # Display layer 1 immediately, deeper layers on request
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DisclosureLayer:
    """One layer of progressively disclosed results."""
    level: int              # 1=headline, 2=summary, 3=full, 4=methods
    title: str
    content: str
    expandable: bool = True
    token_estimate: int = 0  # approximate tokens in this layer

    @property
    def level_name(self) -> str:
        return {1: "Headline", 2: "Summary", 3: "Full Results", 4: "Methods"}.get(self.level, f"Layer {self.level}")


@dataclass
class DisclosedResult:
    """A complete set of disclosure layers for a result."""
    skill_name: str
    layers: list[DisclosureLayer] = field(default_factory=list)
    current_level: int = 1     # which level is currently being shown
    total_records: int = 0

    def get_current(self) -> str:
        """Get content for the current disclosure level."""
        for layer in self.layers:
            if layer.level == self.current_level:
                return layer.content
        return self.layers[0].content if self.layers else ""

    def get_up_to(self, level: int) -> str:
        """Get all content up to and including the specified level."""
        parts = []
        for layer in sorted(self.layers, key=lambda l: l.level):
            if layer.level <= level:
                parts.append(f"### {layer.title}\n{layer.content}")
        return "\n\n".join(parts)

    def has_more(self) -> bool:
        """Whether there are deeper layers available."""
        return self.current_level < max(l.level for l in self.layers) if self.layers else False


class ProgressivePresenter:
    """Format analysis results for progressive disclosure.

    Automatically detects result type and formats into appropriate layers.
    """

    def __init__(self, top_n_summary: int = 10, headline_n: int = 3) -> None:
        self.top_n_summary = top_n_summary
        self.headline_n = headline_n

    def format_auto(self, skill_name: str, result: dict[str, Any]) -> DisclosedResult:
        """Auto-detect result type and format appropriately."""
        if "phewas" in skill_name.lower() or "associations" in result:
            return self.format_phewas(result)
        elif "train" in skill_name.lower() or "auc" in result or "auc_mean" in result:
            return self.format_model_result(skill_name, result)
        elif "gwas" in skill_name.lower() or "hits" in result:
            return self.format_gwas(result)
        elif "survival" in skill_name.lower() or "hazard_ratio" in result:
            return self.format_survival(result)
        else:
            return self.format_generic(skill_name, result)

    def format_phewas(self, result: dict[str, Any]) -> DisclosedResult:
        """Format PheWAS results in progressive layers."""
        associations = result.get("associations", result.get("results", []))
        if not isinstance(associations, list):
            return self.format_generic("phewas", result)

        n_total = len(associations)
        n_significant = sum(
            1 for a in associations
            if isinstance(a, dict) and a.get("p_value", 1.0) < 5e-8
        )

        # Sort by p-value
        sorted_assoc = sorted(
            [a for a in associations if isinstance(a, dict)],
            key=lambda x: x.get("p_value", 1.0),
        )

        # Layer 1: Headline
        if n_significant > 0:
            top = sorted_assoc[0]
            headline = (
                f"PheWAS identified **{n_significant} genome-wide significant associations** "
                f"out of {n_total} phenotypes tested. "
                f"Strongest: {top.get('phenotype', 'unknown')} (p={top.get('p_value', 0):.2e})"
            )
        else:
            headline = f"PheWAS tested {n_total} phenotypes — no genome-wide significant associations (p < 5×10⁻⁸)"

        # Layer 2: Summary table (top N)
        summary_lines = ["| Rank | Phenotype | p-value | OR | Direction |", "|------|-----------|---------|----|-----------"]
        for i, a in enumerate(sorted_assoc[:self.top_n_summary], 1):
            pheno = a.get("phenotype", "—")[:30]
            p = f"{a.get('p_value', 0):.2e}"
            odds = f"{a.get('odds_ratio', a.get('or', '—')):.3f}" if isinstance(a.get("odds_ratio", a.get("or")), (int, float)) else "—"
            direction = "↑" if a.get("beta", a.get("odds_ratio", 1)) > 1 else "↓"
            summary_lines.append(f"| {i} | {pheno} | {p} | {odds} | {direction} |")
        summary = "\n".join(summary_lines)

        # Layer 3: Full results
        full_lines = [f"Complete results for all {n_total} associations:\n"]
        for i, a in enumerate(sorted_assoc, 1):
            full_lines.append(f"{i}. {a.get('phenotype', '?')} — p={a.get('p_value', 0):.2e}, OR={a.get('odds_ratio', '?')}")
        full = "\n".join(full_lines[:200])  # cap at 200 to prevent overflow
        if n_total > 200:
            full += f"\n... and {n_total - 200} more (export for full table)"

        # Layer 4: Methods
        methods = (
            f"**PheWAS Parameters**: Tested {n_total} phenotypes. "
            f"Significance threshold: p < 5×10⁻⁸ (genome-wide). "
            f"Multiple testing correction: Bonferroni (α = {5e-8:.2e}). "
            f"Model: logistic regression adjusted for age, sex, principal components."
        )

        return DisclosedResult(
            skill_name="phewas",
            layers=[
                DisclosureLayer(1, "Key Finding", headline, token_estimate=len(headline) // 4),
                DisclosureLayer(2, f"Top {min(self.top_n_summary, n_total)} Associations", summary, token_estimate=len(summary) // 4),
                DisclosureLayer(3, "All Associations", full, token_estimate=len(full) // 4),
                DisclosureLayer(4, "Methodology", methods, token_estimate=len(methods) // 4),
            ],
            total_records=n_total,
        )

    def format_model_result(self, skill_name: str, result: dict[str, Any]) -> DisclosedResult:
        """Format ML model training/evaluation results."""
        auc = result.get("auc") or result.get("auc_mean") or result.get("roc_auc", 0)
        ci_low = result.get("auc_ci_low") or result.get("auc_lower", None)
        ci_high = result.get("auc_ci_high") or result.get("auc_upper", None)
        n_features = result.get("n_features", 0)
        model_type = result.get("model_type", result.get("model", "unknown"))

        # Layer 1: Headline
        ci_str = f" (95% CI: {ci_low:.3f}-{ci_high:.3f})" if ci_low is not None and ci_high is not None else ""
        headline = f"Model achieves **AUC = {auc:.3f}**{ci_str} using {n_features} features ({model_type})"

        # Layer 2: Metrics summary
        metrics_lines = ["| Metric | Value |", "|--------|-------|"]
        for key in ("auc", "auc_mean", "accuracy", "precision", "recall", "f1_score", "specificity", "sensitivity"):
            if key in result:
                metrics_lines.append(f"| {key.replace('_', ' ').title()} | {result[key]:.4f} |")
        if "n_cases" in result:
            metrics_lines.append(f"| Cases | {result['n_cases']:,} |")
        if "n_controls" in result:
            metrics_lines.append(f"| Controls | {result['n_controls']:,} |")
        summary = "\n".join(metrics_lines)

        # Feature importance (if available)
        features = result.get("feature_importance", result.get("top_features", []))
        if features:
            summary += "\n\n**Top Features:**\n"
            for i, feat in enumerate(features[:5], 1):
                if isinstance(feat, dict):
                    summary += f"{i}. {feat.get('name', '?')} (importance: {feat.get('importance', 0):.4f})\n"
                elif isinstance(feat, (list, tuple)) and len(feat) >= 2:
                    summary += f"{i}. {feat[0]} (importance: {feat[1]:.4f})\n"

        # Layer 3: Full results
        full = json.dumps(result, indent=2, default=str) if isinstance(result, dict) else str(result)

        # Layer 4: Methods
        methods = (
            f"**Model**: {model_type}\n"
            f"**Features**: {n_features}\n"
            f"**Cross-validation**: {result.get('cv_folds', 5)}-fold\n"
            f"**Split strategy**: {result.get('split', 'stratified')}\n"
        )

        return DisclosedResult(
            skill_name=skill_name,
            layers=[
                DisclosureLayer(1, "Model Performance", headline, token_estimate=len(headline) // 4),
                DisclosureLayer(2, "Detailed Metrics", summary, token_estimate=len(summary) // 4),
                DisclosureLayer(3, "Complete Output", full, token_estimate=len(full) // 4),
                DisclosureLayer(4, "Training Parameters", methods, token_estimate=len(methods) // 4),
            ],
        )

    def format_gwas(self, result: dict[str, Any]) -> DisclosedResult:
        """Format GWAS results in layers."""
        hits = result.get("hits", result.get("significant_variants", []))
        n_tested = result.get("n_variants_tested", result.get("n_tested", "unknown"))

        # Layer 1
        if hits:
            top_hit = hits[0] if isinstance(hits[0], dict) else {"variant": str(hits[0])}
            headline = (
                f"GWAS identified **{len(hits)} significant loci** from {n_tested} variants tested. "
                f"Top hit: {top_hit.get('variant', top_hit.get('rsid', '?'))} (p={top_hit.get('p_value', '?')})"
            )
        else:
            headline = f"GWAS tested {n_tested} variants — no genome-wide significant hits"

        # Layer 2
        if hits:
            lines = ["| Variant | CHR | Position | p-value | Gene |", "|---------|-----|----------|---------|------|"]
            for h in hits[:self.top_n_summary]:
                if isinstance(h, dict):
                    lines.append(
                        f"| {h.get('rsid', h.get('variant', '?'))} | "
                        f"{h.get('chr', '?')} | {h.get('pos', '?')} | "
                        f"{h.get('p_value', 0):.2e} | {h.get('gene', '?')} |"
                    )
            summary = "\n".join(lines)
        else:
            summary = "No significant variants to display."

        return DisclosedResult(
            skill_name="gwas",
            layers=[
                DisclosureLayer(1, "GWAS Summary", headline),
                DisclosureLayer(2, f"Top Loci", summary),
                DisclosureLayer(3, "All Hits", json.dumps(hits, default=str, indent=2)[:5000]),
                DisclosureLayer(4, "Methods", f"Variants tested: {n_tested}. Threshold: 5×10⁻⁸."),
            ],
            total_records=len(hits),
        )

    def format_survival(self, result: dict[str, Any]) -> DisclosedResult:
        """Format survival analysis results."""
        hr = result.get("hazard_ratio", result.get("hr", "?"))
        ci_low = result.get("hr_ci_low", result.get("ci_lower", None))
        ci_high = result.get("hr_ci_high", result.get("ci_upper", None))
        p = result.get("p_value", "?")

        ci_str = f" (95% CI: {ci_low:.2f}-{ci_high:.2f})" if ci_low is not None and ci_high is not None else ""
        if isinstance(hr, (int, float)) and isinstance(p, float):
            headline = f"Hazard Ratio = **{hr:.2f}**{ci_str}, p = {p:.2e}"
        else:
            headline = f"HR = {hr}{ci_str}"

        return DisclosedResult(
            skill_name="survival",
            layers=[
                DisclosureLayer(1, "Survival Analysis", headline),
                DisclosureLayer(2, "Details", json.dumps(result, default=str, indent=2)[:2000]),
            ],
        )

    def format_generic(self, skill_name: str, result: dict[str, Any]) -> DisclosedResult:
        """Generic formatting for any skill result."""
        # Create a simple headline from first few keys
        key_items = [(k, v) for k, v in result.items() if not k.startswith("_")][:3]
        headline_parts = [f"{k}={v}" for k, v in key_items if not isinstance(v, (list, dict))]
        headline = f"{skill_name}: " + ", ".join(headline_parts) if headline_parts else f"{skill_name} completed"

        return DisclosedResult(
            skill_name=skill_name,
            layers=[
                DisclosureLayer(1, "Result", headline),
                DisclosureLayer(2, "Details", json.dumps(result, default=str, indent=2)[:3000]),
            ],
        )


