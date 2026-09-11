# V1 分层专家架构

## 目标

V1 在不引入自由式 Multi-Agent 协商的前提下，使用模型 Function Call 声明语义意图，再由
Python Policy 校验计划并路由到任务专家。模型规划失败时回退本地规则 Planner；生产代码
不按题目 ID 分支，历史 Bad Case 只作为回归测试。

## 执行链

```text
QuestionRecord
  -> FunctionCallingIntentPlanner（无图片，只声明计划）
  -> Python Plan Policy / AgentRouter（失败时规则 Planner 兜底）
  -> Extract / Compute / Structure / Visual Agent
  -> direct_answer 或 inspect_table_region tool_call
  -> 共享 Locator / high-resolution ROI / OCR
  -> EvidenceResponse
  -> Operation Executor（需要计算时）
  -> Evidence Soft Validation -> 记录风险并保留原答案
  -> Answer Projection
  -> Normalizer / Contract Validator
```

Agent 可主动调用共享工具，执行失败时也可被动触发同一恢复能力：

```text
Operation/Argument error -> Operation Repair -> Executor
Insufficient/visual error -> Locator -> high-resolution ROI -> OCR candidate -> Specialist retry
Evidence risk -> record warning -> keep original answer
Structure geometry error -> deterministic Structure Repair -> schema validation
```

所有恢复都有次数上限。软验证器只返回风险码，不具有清空答案或拒绝发布的权限。

## 专家职责

- Extract：单值直接答案；多字段、行列和区域抽取按需组装结果。
- Compute：提取操作数，时间差、平均值、百分点、极值和多步 DAG 由 Python 执行。
- Structure：完整/局部结构恢复和结构数值。
- Visual Attribute：颜色、方向、布局、图片或列是否存在等视觉属性。

## Evidence 与答案投影

Evidence 支持稳定 `id`、字段角色、实体、指标、值类型、页码、归一化 bbox 和原始文本。
Operation 优先通过 `evidence_id` 引用，也可按索引引用；两种方式统一支持值、行列头、
单位、`entity`、`metric` 和 `source_text`。`argmax/argmin` 返回完整记录，Answer
Projection 支持 identity、path 和 fields 三种模式，从完整结果中按需选择最终答案。

## 高清裁剪与 OCR

Locator 返回 0 到 1 的归一化 bbox。PDF 区域从原始文档按 `roi_dpi` 重新渲染，图片区域
从原始像素裁剪，不对低清缓存图直接放大。OCR 是可插拔候选生成器；OCR 文本不会直接
成为答案，专家必须结合候选区域图片进行绑定。

## Function Calling 边界

Intent Planner 必须通过 `create_task_plan` Function Call 返回结构化计划，不能回答题目。
Extract、Compute 和 Structure Agent 可以使用 OpenAI 兼容协议的原生 `tool_calls` 请求
`inspect_table_region`。调用参数先经 Pydantic 校验，Router 未授权的 Agent 看不到该工具；
每题调用次数受配置限制。工具完成后，Agent 必须基于高清区域和 OCR 返回最终候选，不能
形成无限调用循环。工具失败时重新要求最佳回答，已有候选始终不会因软验证被清空。
