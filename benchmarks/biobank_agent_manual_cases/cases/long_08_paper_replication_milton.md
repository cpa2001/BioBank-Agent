# long_08_paper_replication_milton

## Purpose

测试论文复现能力：读取/检索论文、抽取可复现目标、映射到 UKB-only 可行分析、模型训练与差异解释。

## Main Prompt

```text
Reproduce the closest feasible UKB-only slice of the MILTON-style disease prediction paper with DOI 10.1038/s41588-024-01898-1. If the exact paper features are unavailable, choose a defensible E11 Type 2 Diabetes biomarker prediction approximation. Read or fetch the paper when possible, do related deep research, map paper methods to local UKB fields, train an auto-selected model, evaluate calibration and feature importance, compare what is and is not replicated, and produce a technical plus Nature-style dual report.
```

## Commands

```text
/plan Reproduce the closest feasible UKB-only slice of the MILTON-style disease prediction paper with DOI 10.1038/s41588-024-01898-1. If the exact paper features are unavailable, choose a defensible E11 Type 2 Diabetes biomarker prediction approximation. Read or fetch the paper when possible, do related deep research, map paper methods to local UKB fields, train an auto-selected model, evaluate calibration and feature importance, compare what is and is not replicated, and produce a technical plus Nature-style dual report.
/plan-approve
/codex-check paper-method mapping, approximation validity, model diagnostics, and report quality
/claude-check replication gaps and unsupported claims
```

## Expected Tool Path

- `fetch_paper`, `read_paper`, `read_pdf`, or a graceful fallback if unavailable.
- `deep_research`
- `field_search`
- `cohort_summary`
- `missing_data`
- `train_model(model_type="auto")`
- `evaluate_model`
- `calibration`
- `feature_importance`
- `statistical_review`
- `safety_check`
- `world_model_audit`
- `generate_report(format="dual")`

## Scoring - 100 pts

- 20: Correctly identifies paper target and feasible local approximation.
- 15: Does not pretend exact replication if features/data differ.
- 20: Trains and evaluates a real model branch when data support it.
- 15: Compares paper methods/results to local UKB output.
- 15: Includes guardrails and non-causal limits.
- 15: Report is polished and useful to a researcher.

## Red Flags

- Paper is cited but not read/fetched or acknowledged as unavailable.
- Agent claims full replication without evidence.
- No model branch.
- Report hides differences between paper and local approximation.

