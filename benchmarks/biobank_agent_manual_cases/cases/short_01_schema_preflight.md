# short_01_schema_preflight

## Purpose

测试 agent 是否能在正式执行前暴露真实 skill schema、项目文档入口和字段搜索能力。这个 case 不追求最终报告，主要看交互和 schema 可靠性。

## Commands

```text
/status
/skills
/mcp-list
List project documentation that is useful for planning a UKB Type 2 Diabetes analysis, then search the UKB field catalogue for BMI, HbA1c, glucose, blood pressure, and Type 2 Diabetes fields. Do not run modelling yet. Summarize which exact skills and argument names should be used in a later plan.
```

## Expected Tool Path

- `project_doc` or equivalent project documentation lookup.
- `field_search(query, limit)` with multiple concrete queries or robust query expansion.
- No model training.
- No final report required.

## Scoring - 100 pts

- 25: Lists relevant docs or project capabilities without guessing nonexistent APIs.
- 25: Field search returns nonzero results for common cardiometabolic terms.
- 20: States real schema names for later use, especially `field_search(query)`, `train_model(icd10_code, model_type)`, `generate_report(title, format, output_dir)`.
- 15: `/mcp-list` reports any configured MCP servers without crashing.
- 15: No schema errors, traceback, or hallucinated tool names.

## Red Flags

- `field_search` returns `total=0` for all broad cardiometabolic terms.
- Agent suggests invalid args such as `sections`, `term`, `docs`, or `target`.
- Agent starts a long plan or training without being asked.

