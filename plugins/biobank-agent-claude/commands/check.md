---
description: Review Biobank Agent execution, tests, and report quality without editing files
argument-hint: "[focus]"
allowed-tools: Read, Glob, Grep, Bash(python:*), Bash(pytest:*), Bash(git:*)
---

Review the current Biobank Agent work read-only.

Focus:
$ARGUMENTS

Required flow:

1. Run `git status --short` and the bridge status command.
2. Inspect the changed files or generated report path relevant to the focus.
3. Run the narrowest relevant tests if they are cheap; otherwise state exactly
   which tests should be run and why.
4. Lead with findings ordered by severity, using file:line references for code
   issues.

Review dimensions:

- agent-callable skill integration,
- statistical validity and impossible metric/cohort checks,
- privacy and small-count safeguards,
- report structure, scientific style, caveats, and reproducibility,
- curated Markdown and plugin docs remain visible through `project_doc`,
- failure behavior when external CLIs or optional dependencies are absent.

Do not edit files.
