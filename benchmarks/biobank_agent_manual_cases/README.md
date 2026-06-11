# Biobank Agent Manual Benchmark Cases

这个目录用于人工测试 Biobank Agent 的真实交互能力。每个 case 都给出：

- 可以直接粘贴到 `biobank` REPL 的输入。
- 建议的执行步骤。
- 期望调用的 tools/skills/plugins。
- 必须检查的 plan/report artifacts。
- 100 分制采分点和明确 fail 条件。

它补充自动测试，不替代 pytest。这里测试的是“像人一样使用 agent”的端到端表现：规划质量、schema 有效性、自修复、模型训练、论文复现、研究现状分析、多模型 debate、trajectory feasibility、最终报告质量。

## 快速开始

```bash
cd /Users/chenpengan/Projects/CUHK/UKB_agent
python -m biobank_agent.cli
```

在 REPL 里按每个 case 的 `Commands` 执行。最常见流程是：

```text
/plan <case prompt>
/plan-approve
```

`/status`、`/skills`、`/plans`、`/routing-status` 只是诊断命令，不是主流程必需步骤。`/plan` 会自动显示计划阶段进度、merge、validation 和 review 状态。

如果执行完成后 CLI 弹出 review hook 选项，按提示处理；通常选 `N` 跳过即可。

## 目录结构

```text
benchmarks/biobank_agent_manual_cases/
  README.md
  RUNBOOK.md
  CASE_MATRIX.tsv
  cases/
    short_01_schema_preflight.md
    short_02_e11_dual_report_smoke.md
    short_03_deep_research_brief.md
    medium_04_auto_model_training.md
    medium_05_trajectory_feasibility.md
    medium_06_multimodel_debate_planning.md
    medium_07_idea_to_study_spec.md
    long_08_paper_replication_milton.md
    long_09_research_landscape_gap.md
    long_10_adaptive_repair_custom_skill.md
    grand_challenge_99_general_agent_showcase.md
  ground_truth/
    TRAJECTORY_GROUND_TRUTH.md
  scoring/
    COMMON_RUBRIC.md
    REPORT_CHECKLIST.md
  templates/
    MANUAL_SCORE_SHEET.md
  prompts/
    live_runner_prompts.txt
```

## 统一通过标准

一个 case 只有在这些条件同时满足时才算通过：

- `plans/<timestamp>.md` 中计划状态为 `DONE`，或如果 `PAUSED`，暂停原因清晰且合理。
- 所有执行过的 step 都使用真实 skill schema，没有 `unexpected keyword argument`。
- 如果 prompt 要求报告，`reports/<timestamp>/` 必须有完整 report artifacts。
- 任何 skill 返回 `error`、`requires_repair` 或 report artifact 缺失时，agent 不能显示假成功。
- 最终报告不能包含作者/机构/AI 工具痕迹，不能出现 unsupported causal/clinical deployment claims。

完整报告 artifacts：

```text
report.md
report_technical.md
report_nature.md
_report_with_css.md
_report_nature_with_css.md
report.html
report_nature.html
```

## Case 选择建议

- 快速冒烟：`short_01`, `short_02`
- 模型训练能力：`medium_04`
- trajectory/HealthFormer 边界：`medium_05`
- 多模型规划：`medium_06`
- 论文复现：`long_08`
- 研究现状与 idea 解析：`medium_07`, `long_09`
- 自修复和 skill 扩展：`long_10`
- 最能展示 agent 综合能力：`grand_challenge_99`

## 可选：用 live-test runner 批量跑

如果要把这些人工 case 交给 pseudo-terminal runner 做并发长测，可以用本地脚本：

```bash
python scripts/run_live_test.py \
  --workers 4 \
  --timeout-min 90 \
  --prompt-file benchmarks/biobank_agent_manual_cases/prompts/live_runner_prompts.txt \
  --output-root reports/biobank_live_tests/manual_benchmark
```
