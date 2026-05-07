# Biobank Agent v2.0 Demo Brief (中英双语)

> Last updated: 2026-04-23  
> Verified test status: `374 passed, 9 skipped`

---

## 1) 当前系统做到了什么 / What The Agent Can Do Now

### 中文
Biobank Agent 现在已经从“会聊天的工具调用器”升级为“可审计的科研执行系统”，核心能力包括：

1. 原生多模型编排：支持复杂任务自动路由到 `single / ensemble / debate / supervisor`。  
2. 分歧强制裁决：多模型意见不一致时，会强制触发工具执行裁决，而不是直接口头共识。  
3. 证据链可追溯：每个关键 claim 可以通过 Action Graph 回溯到执行记录与证据节点。  
4. 科研安全闸门：对高风险结论启用统计与证据约束，无执行证据时阻断“最终科研结论”。  
5. gate fail 自动回滚：策略失败时自动回退到 `ensemble` 再到 `single`，提升稳定性。  
6. CLI 交互升级：启动面板、命令面板、slash 自动补全、路由状态与证据查询命令都已可用。  
7. 评测与门禁：支持 `research_eval_v1` 观测指标与 gate 判定，可做 baseline vs mas_v2 对照。

### English
Biobank Agent now behaves like an auditable scientific execution system rather than a plain chat wrapper:

1. Native multi-model orchestration (`single / ensemble / debate / supervisor`).  
2. Forced adjudication on disagreement (tool-grounded, not just text consensus).  
3. Claim-level traceability via Action Graph.  
4. Scientific safety gate that blocks high-risk conclusions without execution evidence.  
5. Automatic rollback on gate failure (`ensemble` then `single`).  
6. Upgraded CLI UX: startup dashboard, command palette, slash autocomplete, routing/evidence inspection.  
7. Evaluation harness with reliability gates (`research_eval_v1`, baseline vs mas_v2).

---

## 2) 重点亮点 / Key Highlights

### 中文
1. 不再“只看模型回答”，而是要求“可执行证据”。  
2. 把“多智能体讨论”从展示效果，升级为“有裁决约束的硬流程”。  
3. 把可靠性指标（证据覆盖率、统计违规率、错误共识率）纳入日常评测。  
4. 交互层可用性提升，便于现场演示与导师评审。

### English
1. Execution evidence is now first-class, not optional.  
2. Multi-agent discussion is protocol-driven, not just cosmetic.  
3. Reliability metrics are measurable and gateable.  
4. CLI is now presentation-friendly for live demos and supervisor reviews.

---

## 3) 问题与修复对照 / Problem → Fix Mapping

| 问题（中文） | Fix (English) | 结果 |
|---|---|---|
| `/help` 富文本标签在终端里显示为原始文本 | Reworked help rendering with Rich tables/panels instead of raw style tags | 命令面板可读性显著提升 |
| CLI 交互丑、信息密度低 | Added startup dashboard, hints panel, model/routing summary | 首屏可直接展示系统能力 |
| 输入 `/` 后无联想 | Added slash completer (`Tab`, inline menu, history) | 与现代 CLI 使用习惯一致 |
| 部分 relay 报错：`temperature` deprecated (400) | Added deprecated-parameter auto-adaptation and retry in LLM client | 兼容更多中转站模型实现 |
| 复杂任务多模型偶发卡住 | Added global parallel timeout + cancellation of slow futures | 降低卡死概率 |
| 多模型分歧时可能直接“文字和稀泥” | Forced adjudication tool calls when disagreement occurs | 分歧必须转执行证据裁决 |
| 安全 gate fail 后缺少稳态策略 | Implemented auto-rollback chain to `ensemble/single` | 失败时更稳 |
| claim 结论难追溯 | Added Action Graph + `/evidence <claim_id>` | 可展示 claim→evidence 链路 |
| 缺少统一科研可靠性验收 | Added `research_eval_v1` + reliability dashboard + gate checks | 可做持续回归与 A/B |

---

## 4) 演示前准备 / Demo Prerequisites

### 中文
1. 确认 `.env` 中 `DATA_DIR` 可用。  
2. 确认 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 正确。  
3. 建议启用：`MULTI_MODEL_ENABLED=true`。  
4. 建议设置：`MODEL_POOL` 或 `PREFERRED_MULTI_MODELS`。  
5. 启动：`biobank`。

### English
1. Ensure `DATA_DIR` is valid in `.env`.  
2. Set `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.  
3. Enable `MULTI_MODEL_ENABLED=true`.  
4. Configure `MODEL_POOL` or `PREFERRED_MULTI_MODELS`.  
5. Launch with `biobank`.

---

## 5) 高覆盖 Demo 输入脚本（建议逐条粘贴）  
## Comprehensive Demo Input Script (paste step by step)

> 目标 / Goal: 同时覆盖命令系统、多模型编排、证据链、安全闸门、分析技能、研究技能、报告输出。

```text
/help
/models-pool
/models-available
/strategy auto

/plan Comprehensive cardiovascular risk analysis in UK Biobank with strict evidence tracing and safety guardrails. Build cohorts, run biomarkers + modeling + survival + comorbidity + literature cross-check, and finish with a publication-ready report.
/plan-approve

Please run an end-to-end cardiovascular analysis pipeline for ICD10 I21 and related cardiometabolic risk factors. 
Requirements:
1) Start with prevalence and cohort summary;
2) compare key biomarkers (lipids, glucose, blood pressure related fields);
3) train and evaluate at least one predictive model with calibration and feature importance;
4) run survival analysis if event data are available;
5) run comorbidity/phewas style exploration when feasible;
6) perform statistical_review and safety_check before final conclusion;
7) if model opinions disagree, trigger tool-based adjudication;
8) keep an explicit evidence chain for claims.

/routing-status

Now do a deep literature triangulation on “cardiovascular risk stratification biomarkers in population-scale cohorts”, and align external evidence with available UKB fields.

/routing-status

Please generate final outputs:
1) a technical report;
2) an IMRaD-style paper draft summary;
3) clear limitations and what needs human review.
Also list the generated file paths.

/history
/figures
/models
/routing-status
```

---

## 6) 一条“高质感主提示词”（可单次输入）  
## One High-Impact Single Prompt

```text
Run a full, publication-oriented cardiovascular risk study workflow on UK Biobank data with strict reliability constraints.

Task scope:
- Disease focus: ICD10 I21 (and close cardiovascular context where relevant).
- Steps: prevalence -> cohort construction -> biomarker comparison -> model training/evaluation/calibration -> feature importance -> survival/comorbidity (if data supports) -> literature triangulation.
- Reliability constraints:
  1) no final scientific conclusion without execution-grounded evidence;
  2) if multi-model disagreement appears, force tool-execution adjudication;
  3) perform statistical_review + safety_check before final synthesis.

Deliverables:
1) claim-evidence structured synthesis;
2) publication-ready technical summary with key metrics and uncertainty;
3) generate report artifacts and provide exact file paths;
4) tell me which claims are PARTIAL and why.
```

---

## 7) 演示时可强调的“高级感”点 / What To Highlight During Demo

### 中文
1. `/routing-status` 展示当前策略、分歧状态、claim 与 evidence 计数。  
2. `/evidence <claim_id>` 现场展示 claim 的可追溯证据链。  
3. 若出现分歧或高风险断言，系统会进入 `PARTIAL` 并要求先执行证据工具。  
4. 生成报告后，展示 `report.md / report.html` 路径与图表清单。  
5. 演示后可跑 `biobank eval --suite research_eval_v1 --mode mas_v2 --ab --enforce-gate` 做量化背书。

### English
1. Use `/routing-status` to show strategy/disagreement/claim-evidence counts.  
2. Use `/evidence <claim_id>` to show traceability in real time.  
3. Show that high-risk conclusions are blocked without execution evidence (`PARTIAL`).  
4. Present generated report artifacts (`report.md`, `report.html`) and figure list.  
5. Back it up with benchmark command and gate metrics.

---

## 8) 常见演示风险与应对 / Common Demo Risks and Mitigations

### 中文
1. 中转站偶发 5xx 或某模型临时不可用：提前准备可切换模型，必要时 `/model` 切换。  
2. 数据字段覆盖不全：让 agent 明确输出“可做/不可做边界”，并保留 `PARTIAL`。  
3. 结果过长：先要“结论摘要 + 证据表”，再要完整报告。  
4. 现场网络波动：优先跑本地数据分析链，再补文献检索。

### English
1. Relay/model availability can fluctuate: keep fallback models ready.  
2. Field coverage may vary: enforce explicit boundary reporting (`PARTIAL` if needed).  
3. Response can be long: request concise summary + evidence table first.  
4. If network is unstable: prioritize local analytics pipeline, then literature layer.

---

## 9) 快速验收清单 / Quick Acceptance Checklist

### 中文
1. `/` + `Tab` 命令补全是否正常。  
2. `/help` 是否美观可读。  
3. `/routing-status` 是否出现 strategy/claims/evidence/safety。  
4. `/evidence <claim_id>` 是否能查到链路。  
5. 最终报告是否产出且路径可见。  
6. 若高风险结论无证据，是否被 `PARTIAL` 阻断。

### English
1. Slash autocomplete works (`/` + `Tab`).  
2. `/help` is readable and styled correctly.  
3. `/routing-status` shows strategy/claims/evidence/safety.  
4. `/evidence <claim_id>` resolves the chain.  
5. Report artifacts are generated with visible paths.  
6. High-risk unsupported conclusions are blocked as `PARTIAL`.

