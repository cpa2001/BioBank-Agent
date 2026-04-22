"""Critical thinking skill — structured evaluation of scientific claims.

Implements a 7-step critical evaluation protocol with biobank-specific
red-flag detection for common biases in large cohort studies.
"""

from __future__ import annotations

import logging
from typing import Any

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Biobank-specific red flags
# ---------------------------------------------------------------------------

_BIOBANK_RED_FLAGS: list[dict[str, str]] = [
    {
        "flag": "Healthy volunteer bias",
        "description": (
            "UK Biobank participants are healthier, wealthier, and more educated "
            "than the general UK population. Prevalence estimates and risk associations "
            "may not generalise. Check if authors acknowledge this."
        ),
        "severity": "high",
    },
    {
        "flag": "Survivorship bias",
        "description": (
            "Participants were aged 40-69 at recruitment (2006-2010). Diseases with "
            "high early mortality (e.g. childhood cancers, Type 1 diabetes complications) "
            "are under-represented. Cross-sectional analyses conflate incidence and prevalence."
        ),
        "severity": "high",
    },
    {
        "flag": "Time-varying confounders",
        "description": (
            "Baseline measurements may not reflect values at the time of outcome. "
            "Biomarkers, medications, and lifestyle change over follow-up. Single "
            "time-point exposure measurement introduces regression dilution bias."
        ),
        "severity": "medium",
    },
    {
        "flag": "Collider bias / selection bias",
        "description": (
            "Conditioning on UKB participation can induce collider bias. Associations "
            "observed within UKB may be distorted. Check for index-event bias in "
            "case-only analyses."
        ),
        "severity": "medium",
    },
    {
        "flag": "Multiple testing without correction",
        "description": (
            "With thousands of available fields and ICD codes, running multiple "
            "associations without Bonferroni, FDR, or permutation correction "
            "inflates false-positive rates. PheWAS especially vulnerable."
        ),
        "severity": "high",
    },
    {
        "flag": "Self-reported data reliability",
        "description": (
            "Many UKB fields (medications, diagnoses, lifestyle) are self-reported "
            "at a touchscreen. Misclassification is common, especially for sensitive "
            "topics (alcohol intake, mental health). Check validation against hospital records."
        ),
        "severity": "medium",
    },
    {
        "flag": "Ethnicity stratification",
        "description": (
            "UKB is ~94% White British. GWAS and PRS derived predominantly from "
            "this group may not transfer to other ancestries. Check if authors "
            "restrict to or stratify by genetic ancestry."
        ),
        "severity": "medium",
    },
    {
        "flag": "Winner's curse in discovery",
        "description": (
            "Initial effect sizes from discovery analyses are typically inflated. "
            "Check for independent replication or split-sample validation."
        ),
        "severity": "medium",
    },
    {
        "flag": "Immortal time bias",
        "description": (
            "If exposure classification requires survival to a certain point "
            "(e.g. 'started statin after recruitment'), the time before exposure "
            "is 'immortal' and must be handled correctly in survival analyses."
        ),
        "severity": "high",
    },
    {
        "flag": "Reverse causation",
        "description": (
            "Baseline biomarker levels may already be affected by sub-clinical disease. "
            "Sensitivity analyses excluding early events (e.g. first 2 years of follow-up) "
            "are essential."
        ),
        "severity": "medium",
    },
]


# ---------------------------------------------------------------------------
# Evaluation framework
# ---------------------------------------------------------------------------

def _build_evaluation_framework(claim: str, context: str) -> str:
    """Build the 7-step critical evaluation template."""
    lines = [
        "# Critical Evaluation Framework",
        "",
        "## Claim Under Evaluation",
        "",
        f"> {claim}",
        "",
    ]
    if context:
        lines.extend([
            "## Additional Context",
            "",
            f"{context}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Step 1: Restate the Claim Precisely",
        "",
        "- What exactly is being claimed?",
        "- What is the population, exposure, comparator, and outcome (PECO)?",
        "- Is the claim causal, associational, or predictive?",
        "- What would the null hypothesis be?",
        "",
        "## Step 2: Test Story Logic",
        "",
        "- Does the biological mechanism make sense?",
        "- Is there a plausible causal pathway?",
        "- Are there known contradictions in the literature?",
        "- Does the timeline of cause and effect hold?",
        "",
        "## Step 3: Test Method-Task Fit",
        "",
        "- Is the study design appropriate for the claim being made?",
        "  - Causal claims need RCT or strong quasi-experimental design (MR, DiD, RDD)",
        "  - Association claims need proper confounder adjustment",
        "  - Prediction claims need out-of-sample validation",
        "- Are the statistical methods appropriate?",
        "- Is the sample size adequate for the effect size claimed?",
        "",
        "## Step 4: Test Evidence Quality",
        "",
        "- What is the sample size and statistical power?",
        "- Are confidence intervals reported and interpretable?",
        "- Are P-values used correctly (not just 'significant' / 'not significant')?",
        "- Is there evidence of proper controls?",
        "- Are sensitivity analyses reported?",
        "- Is the data publicly available or reproducible?",
        "",
        "## Step 5: Test for Cherry-Picking",
        "",
        "- Are all pre-specified outcomes reported?",
        "- Are negative or null results mentioned?",
        "- Were subgroup analyses pre-specified or post-hoc?",
        "- Is there evidence of p-hacking (e.g., p-values clustering just below 0.05)?",
        "- Do the authors cite a pre-registration or analysis plan?",
        "",
        "## Step 6: Discount Fancy Methodology",
        "",
        "- Does the methodological complexity add genuine value?",
        "- Could a simpler method answer the same question?",
        "- Is the method validated in this specific context?",
        "- Is 'machine learning' or 'AI' being used where logistic regression would suffice?",
        "- Beware: complexity can obscure rather than illuminate.",
        "",
        "## Step 7: Proportionate Conclusion",
        "",
        "- Are the conclusions warranted by the evidence presented?",
        "- Is the language calibrated to the strength of evidence?",
        "  - Association ≠ causation (unless strong design supports it)",
        "  - Single cohort ≠ universal truth",
        "  - Statistical significance ≠ clinical significance",
        "- What would change your mind about this claim?",
        "",
    ])

    return "\n".join(lines)


def _build_red_flags_section() -> str:
    """Build the biobank-specific red flags checklist."""
    lines = [
        "---",
        "",
        "## Biobank-Specific Red Flags Checklist",
        "",
    ]
    for rf in _BIOBANK_RED_FLAGS:
        severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
            rf["severity"], "⚪"
        )
        lines.extend([
            f"### {severity_icon} {rf['flag']} [{rf['severity'].upper()}]",
            "",
            f"{rf['description']}",
            "",
            "- [ ] Addressed by the authors?",
            "- [ ] Relevant to this specific claim?",
            "",
        ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

@skill(
    name="critical_thinking",
    description="Evaluate a scientific claim or method using a structured critical thinking framework. "
                "Checks for logic traps, evaluates evidence quality, and identifies red flags.",
    parameters={
        "claim": {
            "type": "string",
            "description": "The scientific claim or method to evaluate",
        },
        "context": {
            "type": "string",
            "description": "Additional context about the claim",
            "default": "",
        },
    },
    required=["claim"],
)
def critical_thinking(claim: str, context: str = "", *, ctx=None) -> dict:
    """Evaluate a scientific claim using a 7-step critical thinking protocol.

    The evaluation framework includes:

    1. Restate the claim precisely
    2. Test story logic (biological plausibility)
    3. Test method-task fit (appropriate study design?)
    4. Test evidence quality (sample size, CIs, controls)
    5. Test for cherry-picking (selective reporting)
    6. Discount fancy methodology (does complexity add value?)
    7. Proportionate conclusion (are conclusions warranted?)

    Additionally checks for biobank-specific red flags including healthy
    volunteer bias, survivorship bias, and time-varying confounders.

    Returns
    -------
    dict
        ``{"claim": str, "evaluation_framework": str, "red_flags": str,
           "biobank_red_flags": [...], "steps": [...]}``
    """
    # Build the evaluation
    framework = _build_evaluation_framework(claim, context)
    red_flags_section = _build_red_flags_section()

    # Combine into one document
    full_evaluation = framework + "\n" + red_flags_section

    # Summary of steps for structured return
    steps = [
        {"step": 1, "name": "Restate the Claim Precisely", "status": "to_evaluate"},
        {"step": 2, "name": "Test Story Logic", "status": "to_evaluate"},
        {"step": 3, "name": "Test Method-Task Fit", "status": "to_evaluate"},
        {"step": 4, "name": "Test Evidence Quality", "status": "to_evaluate"},
        {"step": 5, "name": "Test for Cherry-Picking", "status": "to_evaluate"},
        {"step": 6, "name": "Discount Fancy Methodology", "status": "to_evaluate"},
        {"step": 7, "name": "Proportionate Conclusion", "status": "to_evaluate"},
    ]

    return {
        "claim": claim,
        "context": context,
        "evaluation_framework": full_evaluation,
        "red_flags_checklist": [
            {"flag": rf["flag"], "severity": rf["severity"]}
            for rf in _BIOBANK_RED_FLAGS
        ],
        "steps": steps,
        "n_steps": 7,
        "n_red_flags": len(_BIOBANK_RED_FLAGS),
        "instruction": (
            "Use the evaluation framework above to critically analyse the claim. "
            "For each step, provide a concrete assessment based on the available evidence. "
            "Check each biobank red flag for relevance and flag any that apply."
        ),
    }
