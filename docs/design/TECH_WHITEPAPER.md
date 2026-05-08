# Biobank Agent 技术白皮书（落地能力 + Rosalind+ 路线图）

**版本**: v1.0  
**日期**: 2026-04-23  
**受众**: 技术导师 / 科研评审  
**口径**: 已落地能力为主，路线图为辅（工程可执行、可审计）

---

## 一页式结论

Biobank Agent 当前已从“通用对话+工具调用”升级为“**可执行裁决、可证据追溯、可门禁验收**”的科研 Agent 系统。系统核心不再是模型文风，而是三层闭环：

1. **编排层**: 原生多模型路由与分歧处理（`single / ensemble / debate / supervisor`）。  
2. **证据层**: Claim-Evidence 结构化追溯（Action Graph）。  
3. **可靠性层**: 科研安全闸门 + 回滚策略 + 评测门禁（north-star metrics）。

当前上限：在 UKB 垂直科研任务上已具备 Rosalind-like 的工程骨架能力（闭环执行、证据链、可靠性门禁）。  
当前短板：跨模态图检索深度、红蓝对抗学习闭环、长期自进化路由器仍在工程化推进中。  
下一阶段目标：用 8 周任务包把“多模型讨论”升级为“执行可裁决的对抗式科研操作系统”。

---

## 已落地能力（硬证据）

### 核心能力五条（机制-触发条件-输出结构-失败处理）

| 能力 | 机制 | 触发条件 | 输出结构 | 失败处理 |
|---|---|---|---|---|
| 原生多模型编排 | `orchestrator.route()` + complexity classifier 动态选择 `single/ensemble/debate/supervisor` | 复杂度评分或策略强制 | `OrchestrationResult`（`final_answer`, `claims[]`, `evidence_links[]`, `debate_trace`, `safety_status`） | 失败或风险状态进入 gate 与回滚链 |
| 分歧强制裁决 | `debate` 中匿名初票 + 分歧检测 + 强制 adjudication tool calls | 投票分歧且 judge 无工具执行证据 | `debate_trace.disagreement`, `forced_adjudication=true`, 附带 tool calls | 未执行工具则不允许最终科研结论放行 |
| 证据链可追溯 | Action Graph（SQLite）记录 claim / evidence / query / result / figure 软连接 | 产生 claim 或工具执行记录 | `retrieve_claim_evidence`, `explain_claim`, CLI `/evidence <claim_id>` | 无证据时返回显式缺失，触发 PARTIAL |
| 科研安全闸门 | `_apply_execution_judge()` 对高风险断言做统计与证据门控 | 高风险语义、分歧、样本量/混杂因子风险 | `safety_status=PASS\|PARTIAL` + gate reason | 无执行证据时硬阻断科研结论（Adjudication Required） |
| gate fail 自动回滚 | `_rollback_on_gate_fail()` 自动尝试 `ensemble -> single` | 非 PASS 且无可执行 tool_call 放行条件 | `debate_trace.rollback_chain` 记录回滚路径 | 候选策略全部失败则返回最佳可得结果并保留风险状态 |

### 关键实现锚点（代码可审计）

- 编排与回滚: `biobank_agent/orchestrator.py`（`route`, `_rollback_on_gate_fail`）  
- 分歧强制裁决: `biobank_agent/orchestrator.py`（`debate`, `_build_adjudication_tool_calls`）  
- 科研安全闸门: `biobank_agent/orchestrator.py`（`_apply_execution_judge`）  
- Action Graph 接口: `biobank_agent/memory.py`（`upsert_node`, `link_nodes`, `retrieve_claim_evidence`, `explain_claim`）  
- CLI 诊断命令: `biobank_agent/cli.py`（`/routing-status`, `/evidence`, `/models-available`）  
- 评测门禁: `biobank_agent/eval/harness.py`, `biobank_agent/eval/benchmarks.py`（`research_eval_v1`, gate checks）

### 当前测试基线

- 全量回归（2026-04-23）: **`379 passed, 9 skipped`**  
- 说明：新增编排与裁决测试后保持全绿，未引入回归性失败。

---

## 传统 Agent 框架问题对照表（固定 12 行）

| 传统agent框架问题 | biobank agent 解决方案 | 优点 |
|---|---|---|
| 单模型能力天花板 | 原生多模型路由（single/ensemble/debate/supervisor） | 复杂任务成功率与鲁棒性提升 |
| 多智能体“轮流聊天”从众 | 匿名初票 + 分歧触发再检索 | 降低错误共识与附和效应 |
| 分歧无客观裁决 | 强制工具执行裁决（Execution as Judge） | 结论可执行、可验证 |
| 结论不可追溯 | Action Graph claim/evidence 链 | 可审计、可复盘 |
| 高风险断言直接输出 | 科研安全闸门（PARTIAL 阻断） | 降低误导性科学结论 |
| gate fail 无降级策略 | 自动回滚到 `ensemble` 再到 `single` | 系统稳定性增强 |
| 纯 Top-K 向量检索粒度错配 | 双通道检索（语义召回 + 图游走 + 统计充分性重排） | 召回相关性与可解释性更好 |
| 图/表/文本割裂 | 跨模态 grounding MVP 写入 Action Graph | 支持异常信号反向证据检索 |
| 只看“回答流畅度” | `research_eval_v1` + 北极星指标门禁 | 可靠性可量化、可持续回归 |
| 模型可见不等于可用 | `/models-available` + 健康状态/缓存信息 | 运行前可诊断，减少盲测 |
| 多模型并行易卡死 | 并行超时 + 慢请求取消 | 降低长尾阻塞与卡死风险 |
| relay 参数/协议漂移导致 400/503 | 兼容层（deprecated 参数自适应、重试、超时） | 线上兼容性更强 |

---

## Rosalind+/Heisenberg 能力映射（差距透明）

### Rosalind-like（科研闭环工作流）

#### 当前已达成
- 具备从任务意图到工具执行、再到证据链输出的闭环骨架。  
- 具备复杂任务多模型协作与安全闸门约束。  
- 具备评测门禁，不以文风作为主要胜负指标。

#### 当前未达成
- 尚未完成面向生物医学全流程的实验室级（Lab-in-the-loop）自动反馈闭环。  
- 尚未形成大规模长期轨迹驱动的专用 Router/Verifier 微调闭环。

#### 风险点
- 外部 relay 可用性波动会影响多模型协作稳定性。  
- 跨模态证据密度不足时，可能提高 PARTIAL 比例。

#### 补齐动作
- 强化模型可用性探测与动态池管理。  
- 提升 claim-level 证据覆盖率与跨模态反向检索深度。  
- 建立高质量轨迹数据资产，推进轻量 router/verifier 迭代。

### Heisenberg-like（形式化约束优先）

#### 当前已达成
- 分歧裁决优先执行证据，不接受纯自然语言共识。  
- 高风险断言可被 gate 硬阻断并要求执行补证。

#### 当前未达成
- 尚未将所有关键科学子任务统一到严格形式化规约（JSON schema + programmatic checks）下。  
- 尚未实现跨任务全局约束求解器式的统一裁决。

#### 风险点
- 局部工具结果质量不稳会放大裁决链误差。  
- 形式化不足的步骤仍依赖模型表达质量。

#### 补齐动作
- 扩展结构化协议覆盖面（任务分解、反驳、合并、裁决）。  
- 增强 verdict 与统计审查模块的自动一致性校验。

---

## 8 周 Rosalind+ 工程任务包（可执行）

### 北极星指标（必须达标）

- `evidence_coverage >= 95%`  
- `stat_guardrail_violation <= 1%`  
- `complex_task_success >= baseline + 15%`  
- `wrong_consensus_rate` 下降 `>= 40%`

### 周计划

| 周期 | 工程任务包 | 关键交付（DoD） |
|---|---|---|
| Week 1 | 评测与观测面板门禁 | `research_eval_v1` 日跑、统一指标报表、自动 gate 判定 |
| Week 2-3 | 红蓝对抗协议 + JSON 通信 + 分歧强制裁决 | 角色职责固定、协议可测、分歧必经执行裁决 |
| Week 4-5 | Action Graph 强化 + claim→evidence 回溯完整性 | claim 证据链完整率提升、回溯接口稳定 |
| Week 6 | 跨模态 grounding 与降级路径 | 图/表异常写图成功，缺模态时自动降级可解释 |
| Week 7 | 生产级科研安全闸门默认开启 | 样本量、混杂因子、多重校正、证据链完整性全开启 |
| Week 8 | A/B 与复现实验 | 达标发布稳定版；不达标进入受限预览并回滚策略 |

---

## 公共接口与命令（文稿必须点名）

### Public APIs / Types

- `OrchestrationResult`  
  - `final_answer`  
  - `claims[]`  
  - `evidence_links[]`  
  - `debate_trace`  
  - `safety_status`

- Action Graph API  
  - `upsert_node`  
  - `link_nodes`  
  - `retrieve_claim_evidence`  
  - `explain_claim`

### CLI / Eval

- CLI: `/routing-status`, `/evidence <claim_id>`, `/models-available`  
- Eval: `biobank eval --suite research_eval_v1 --mode baseline|mas_v2 [--ab] [--enforce-gate]`

---

## 文稿质量与技术准确性验收

1. **事实一致性检查**: 每条“已实现”陈述必须能映射到当前代码行为。  
2. **口径边界检查**: 已落地能力与路线图能力严格分层，禁止混写。  
3. **指标完整性检查**: 北极星四指标均出现且带阈值。  
4. **演示可执行性检查**: 命令可复现，输出路径可见。  
5. **语言质量检查**: 术语准确、句长可控、无营销化表达。

---

## 可演示命令集与 5 分钟叙事顺序

### 推荐命令集

```bash
/models-available
/routing-status
/evidence <claim_id>
biobank eval --suite research_eval_v1 --mode mas_v2 --ab --enforce-gate
```

### 导师演示 5 分钟叙事顺序

1. **现状**: 展示系统不是单模型对话器，而是可门禁的科研执行系统。  
2. **机制**: 展示多模型路由、分歧强制裁决、安全闸门、回滚机制。  
3. **证据**: 用 `/routing-status` 和 `/evidence` 展示 claim→evidence 链。  
4. **指标**: 展示 `research_eval_v1` 与北极星指标门禁。  
5. **路线图**: 给出 8 周任务包，说明“达标发布 / 不达标受限预览”的发布纪律。

---

## 假设与边界

- 本文“超过 Rosalind”仅定义为 UKB 垂直任务上的可靠性、可追溯性与错误共识控制能力，不作全领域绝对超越声明。  
- 模型策略采用 API 优先，模型组合按可用通道动态选择（优先 `gpt-5.4`，其余模型按可用性降级）。  
- 文稿目标是工程可信与科研可靠，不追求营销化叙事。

