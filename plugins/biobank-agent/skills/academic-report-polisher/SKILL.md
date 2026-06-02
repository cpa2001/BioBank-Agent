---
name: academic-report-polisher
description: Polish AI- or pipeline-generated technical, biomedical, bioinformatics, WGS/GWAS, omics, and academic Markdown reports into concise, human-readable senior-expert reports. Use when asked to improve report readability, remove AI-like wording, rewrite abstract/introduction/methods/results/discussion, integrate separated figures into relevant text, clarify study logic, methods, conclusions, limitations, references, or move execution logs into an appendix.
---

# Academic Report Polisher

Use the BioBank Agent `academic_report_polisher` skill before publishing any
Markdown or HTML report. For generated reports, prefer the built-in
`generate_report(polish=true)` path because it preserves the raw report and
stores a quality JSON next to the final artifact.

## Workflow

1. Read the report, nearby result tables, figures and execution logs.
2. Rebuild the front matter around the research question, cohort, methods,
   results, limitations and conclusion.
3. Keep tool logs, raw dictionaries, run durations and provenance details in an
   execution appendix.
4. Preserve valid paths and do not invent results that are absent from the
   report or result files.
5. Check figure links, main-body machine-output artifacts and claim boundaries
   before returning the report.

## Quality Rules

- Abstract and introduction must be concise.
- Methods should explain what was analyzed, why, and with which thresholds or
  model settings when available.
- Results should interpret evidence strength and separate exploratory findings
  from validated findings.
- Figures should be near the relevant text, numbered sequentially, and captioned.
- WGS/GWAS/omics claims must mention sample size, QC/annotation/association
  boundary, multiple testing or small-sample limits when relevant.
- Predictive models in tiny cohorts should be framed as exploratory diagnostics,
  not deployable biomarkers.
