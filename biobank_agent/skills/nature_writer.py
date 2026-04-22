"""Nature writer skill — write publication-quality text in Nature journal style.

Generates structured scientific text following Nature, Nature Methods, and
Nature Medicine formatting conventions.  Each section type has its own
template with length limits, structural rules, and style guidance.
"""

from __future__ import annotations

import logging
from typing import Any

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style guidelines per journal
# ---------------------------------------------------------------------------

_STYLE_RULES: dict[str, dict[str, Any]] = {
    "nature": {
        "journal": "Nature",
        "abstract_words": 150,
        "abstract_structure": "single paragraph, no headings",
        "intro_refs": "~20-30 references for a full Article",
        "results_style": "integrated with discussion for Letters; separate for Articles",
        "methods_location": "Online Methods section (after references)",
        "tone": "Authoritative but accessible to broad scientific audience",
        "special": (
            "Nature favours bold claims backed by strong evidence. "
            "Opening sentence must be arresting. Avoid jargon where possible."
        ),
    },
    "nature_methods": {
        "journal": "Nature Methods",
        "abstract_words": 200,
        "abstract_structure": "single paragraph, method-focused",
        "intro_refs": "~15-25 references",
        "results_style": "benchmarking and validation emphasis",
        "methods_location": "detailed Methods section with algorithmic descriptions",
        "tone": "Technical precision with clear benchmarks",
        "special": (
            "Emphasise what the method enables that was not possible before. "
            "Include performance benchmarks and comparison to existing methods."
        ),
    },
    "nature_medicine": {
        "journal": "Nature Medicine",
        "abstract_words": 200,
        "abstract_structure": "structured: Background, Methods, Results, Conclusions",
        "intro_refs": "~20-30 references",
        "results_style": "clinical focus with translational implications",
        "methods_location": "detailed Methods with clinical protocol",
        "tone": "Clinical precision with translational impact",
        "special": (
            "Emphasise clinical relevance and patient impact. "
            "Include effect sizes with clinical interpretation. "
            "Address generalisability across populations."
        ),
    },
}


# ---------------------------------------------------------------------------
# Section templates
# ---------------------------------------------------------------------------

def _template_title(content: str, style_rules: dict) -> str:
    return f"""## Writing Instruction: Title

**Journal**: {style_rules['journal']}
**Rules**:
- Maximum 10-12 words (Nature prefers shorter titles)
- No acronyms unless universally known (DNA, RNA, HIV)
- Active, declarative — state the main finding
- Never start with "A study of..." or "Investigation into..."
- Avoid colons; avoid questions unless truly provocative

**Input content/findings**:
{content}

**Write a title following these rules.**
"""


def _template_abstract(content: str, style_rules: dict) -> str:
    return f"""## Writing Instruction: Abstract

**Journal**: {style_rules['journal']}
**Max words**: {style_rules['abstract_words']}
**Structure**: {style_rules['abstract_structure']}

**Style rules**:
- First sentence: broad context (the "So what?" for non-specialists)
- Second sentence: the specific gap or problem
- Then: brief methods summary (one sentence)
- Then: key quantitative results with effect sizes and 95% CI
- Final sentence: implication / significance
- Use exact numbers, never "significant" without P-value
- Active voice preferred; past tense for results
- No references, no acronyms on first use

**Input content/findings**:
{content}

**Write an abstract following these rules.**
"""


def _template_introduction(content: str, style_rules: dict, ctx=None) -> str:
    bank_name = ctx.settings.biobank_name if ctx and hasattr(ctx, "settings") else "Biobank"
    return f"""## Writing Instruction: Introduction

**Journal**: {style_rules['journal']}
**Structure**: Funnel — broad context → specific gap → our approach
**References**: {style_rules['intro_refs']}

**Paragraph plan (3-4 paragraphs)**:
1. **Opening**: Why this topic matters broadly. Start with an arresting fact or
   statement. NEVER open with "In this study" or "Here, we". Do NOT open with
   a cliche about disease burden unless the number is genuinely striking.
2. **What is known**: Summarise key prior work. Be fair and specific — cite
   actual effect sizes and sample sizes where relevant.
3. **The gap**: What is NOT known, or what is wrong with current approaches.
   This gap must logically motivate what comes next.
4. **Our approach**: "Here we [verb]..." in the final paragraph. State the
   study design, dataset ({bank_name}), and preview the key finding.

**Style rules**:
- Sentences: 15-25 words average
- Active voice preferred
- Each paragraph should have a clear topic sentence
- Avoid "Recently, ..." as an opening word

**Tone**: {style_rules['tone']}
**Special**: {style_rules['special']}

**Input content/findings**:
{content}

**Write an introduction following these rules.**
"""


def _template_results(content: str, style_rules: dict) -> str:
    return f"""## Writing Instruction: Results

**Journal**: {style_rules['journal']}
**Style**: {style_rules['results_style']}

**Paragraph structure** (for each result):
1. **Finding first**: State the result immediately (data-first writing).
2. **Evidence**: Report exact numbers: effect size (95% CI), P-value, sample size.
3. **Context**: Brief interpretation — what does this result mean?
4. **Figure reference**: "(Fig. 1a)" at the appropriate point.

**Style rules**:
- Each paragraph = one key finding
- Sub-headings allowed (short, informative)
- Past tense for your results
- Present tense for established facts
- Report exact P-values (P = 0.003), not "P < 0.05"
- For OR/HR: "OR = 1.45 (95% CI, 1.23-1.71; P = 2.3 x 10^-6)"
- For continuous outcomes: "mean difference = 3.2 mmol/L (95% CI, 2.1-4.3)"
- Never say "trend towards significance"
- Distinguish clinical significance from statistical significance

**Input content/findings**:
{content}

**Write a results section following these rules.**
"""


def _template_discussion(content: str, style_rules: dict, ctx=None) -> str:
    bank_caveats = ctx.settings.biobank_caveats if ctx and hasattr(ctx, "settings") else "healthy volunteer bias, selection biases, single time-point measurements"
    return f"""## Writing Instruction: Discussion

**Journal**: {style_rules['journal']}
**Structure**: Mirror the introduction — specific → broad

**Paragraph plan (5-6 paragraphs)**:
1. **Key finding**: Restate the main result in one sentence (do NOT start with
   "In this study, we..."). Open with the finding itself.
2. **Context**: How does this compare to prior work? Agree/disagree with
   existing literature?
3. **Mechanism**: Biological plausibility — why might this be?
4. **Strengths**: Large sample size, prospective design, objective measurements,
   etc. Be specific.
5. **Limitations**: Be honest and specific. Address confounding, generalisability,
   measurement error, reverse causation. Key caveats: {bank_caveats}.
6. **Conclusion**: One paragraph. Clinical/scientific implication. Future directions.
   Do not overstate.

**Style rules**:
- Sentences: 15-25 words average
- Do NOT introduce new data in the Discussion
- Every claim must reference either your results or cited literature
- The limitations section must be substantive, not a token paragraph

**Tone**: {style_rules['tone']}

**Input content/findings**:
{content}

**Write a discussion section following these rules.**
"""


def _template_methods(content: str, style_rules: dict, ctx=None) -> str:
    bank_name = ctx.settings.biobank_name if ctx and hasattr(ctx, "settings") else "Biobank"
    # Conditional note for biobank-specific application numbers
    app_note = (
        f"- For {bank_name}: include application number if applicable"
    )
    return f"""## Writing Instruction: Methods

**Journal**: {style_rules['journal']}
**Location**: {style_rules['methods_location']}

**Required subsections**:
1. **Study population**: Source ({bank_name}), recruitment period, N total,
   inclusion/exclusion criteria, ethical approval + informed consent statement.
2. **Exposure / predictor variables**: How measured, field IDs, units,
   time of measurement.
3. **Outcome variables**: Definition (ICD-10 codes, algorithms), ascertainment
   source (hospital records, death registry, self-report), follow-up period.
4. **Covariates**: List all adjustment variables. Justify each.
5. **Statistical analysis**: Models used, software (R/Python + packages),
   significance threshold, multiple testing correction, sensitivity analyses.
6. **Missing data**: Proportion missing, handling strategy (complete case,
   imputation method).

**Style rules**:
- Past tense throughout
- Sufficient detail for reproduction
- Report software versions
{app_note}
- For genetic data: imputation panel, QC filters, MAF threshold

**Input content/findings**:
{content}

**Write a methods section following these rules.**
"""


_SECTION_BUILDERS: dict[str, Any] = {
    "title": _template_title,
    "abstract": _template_abstract,
    "introduction": _template_introduction,
    "results": _template_results,
    "discussion": _template_discussion,
    "methods": _template_methods,
}


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

@skill(
    name="nature_writer",
    description="Write publication-quality text in Nature journal style. Can write individual "
                "sections (abstract, introduction, results, discussion, methods) or full manuscripts.",
    parameters={
        "section": {
            "type": "string",
            "description": (
                "Section to write: 'abstract', 'introduction', 'results', "
                "'discussion', 'methods', 'title'"
            ),
        },
        "content": {
            "type": "string",
            "description": "Input content/data/findings to write about",
        },
        "style": {
            "type": "string",
            "description": "Journal style: 'nature', 'nature_methods', 'nature_medicine'",
            "default": "nature",
        },
    },
    required=["section", "content"],
)
def nature_writer(section: str, content: str, style: str = "nature", *, ctx=None) -> dict:
    """Generate publication-quality text following Nature journal conventions.

    The function builds a detailed writing-instruction prompt for the
    requested *section* using the specified journal *style*.  The agent LLM
    should then produce the actual text based on this template and the
    supplied *content*.

    Supported sections: ``title``, ``abstract``, ``introduction``,
    ``results``, ``discussion``, ``methods``.

    Returns
    -------
    dict
        ``{"section": str, "text": str, "word_count": int, "style": str}``
    """
    section = section.lower().strip()
    style = style.lower().strip().replace(" ", "_")

    # Validate section
    if section not in _SECTION_BUILDERS:
        return {
            "error": (
                f"Unknown section '{section}'. "
                f"Choose from: {', '.join(sorted(_SECTION_BUILDERS.keys()))}"
            ),
        }

    # Validate style
    if style not in _STYLE_RULES:
        return {
            "error": (
                f"Unknown style '{style}'. "
                f"Choose from: {', '.join(sorted(_STYLE_RULES.keys()))}"
            ),
        }

    style_rules = _STYLE_RULES[style]
    builder = _SECTION_BUILDERS[section]
    writing_prompt = builder(content, style_rules)

    # Build a general style reminder appended to every prompt
    style_reminder = (
        "\n\n## General Style Reminders\n\n"
        "- **Sentence length**: 15-25 words average. Mix short (punch) and long (flow).\n"
        "- **Voice**: Active preferred. 'We found...' not 'It was found...'\n"
        "- **Numbers**: Numerals for >=10 and all measurements. Words for <10 at sentence start.\n"
        "- **Abbreviations**: Spell out on first use. Avoid non-standard abbreviations.\n"
        "- **Hedging**: Use 'suggest', 'indicate', 'is consistent with' — NOT 'prove'.\n"
        "- **References**: [Author et al., Year] format in the text.\n"
        f"- **Journal-specific**: {style_rules['special']}\n"
    )

    full_prompt = writing_prompt + style_reminder

    # Word count of input content
    word_count = len(content.split())

    return {
        "section": section,
        "style": style,
        "journal": style_rules["journal"],
        "writing_prompt": full_prompt,
        "input_content": content,
        "input_word_count": word_count,
        "instruction": (
            f"Use the writing prompt above to generate the {section} section "
            f"in {style_rules['journal']} style. Follow all formatting rules precisely."
        ),
    }
