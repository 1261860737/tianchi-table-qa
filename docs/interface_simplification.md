# 第一阶段：接口简化

基线提交：`eb54102`（当前版本快照，不是 61.8 分实验版源码）。

## 变更边界

- 模型 Planner 只声明题型、专家、原始信息需求和读取需求，不生成操作和答案投影。
- 旧模型响应中的操作、投影不再传给专家；结构题路由等元数据策略仍保留。
- 规则 Planner 及其回退投影保持兼容，但其操作建议不再触发强制重写。
- `OperationSpec` 仍限制操作白名单。引用解析后，由
  `operation_contracts.py` 中的 Pydantic 参数模型检查容器、必填参数和枚举。
- 合同只验证，不转换值或修改列表顺序；数值解析仍由原执行器负责。
  为兼容历史输出，额外辅助参数暂时允许，不采用全局 extra=forbid。
- 参数不匹配统一抛出 `OperationExecutionError`，保留原操作修复路径。

本阶段不修改证据提示词、direct_answer 路径、视觉恢复策略、缺失值和提交兜底策略。
尚未把所有执行失败与视觉恢复完全解耦，不代表完整架构重构已完成。

## 模块与依赖

- `orchestration/planner.py`：缩减模型计划接口。
- `operation_contracts.py`：按操作族定义参数模型。
- `executor.py`：在原子操作执行前统一检查合同。
- `pipeline.py`：移除按计划推荐操作强制重写的检查。
- `tests/test_operation_contracts.py`、`tests/test_planner.py`：覆盖参数边界和兼容行为。

使用项目已有的 Pydantic、pytest，无新增依赖：

```bash
pip install -e '.[dev]'
pytest -q
ruff check src tests
```

## 验证与运行

对本地 `experimental_all_answers_v0.jsonl` 的 908 条 Evidence，分别使用基线和
新执行器离线执行：908 条均执行成功，返回值全部一致。这仅验证执行器兼容性，
不验证新 Planner 的模型输出、全链路答案或评测准确率。

先在线复测历史异常题，日志和提交使用新文件名：

```bash
table-qa run --ids 5,7,8,18,22,26,27,29 --no-resume --workers 4 \
  --log-path outputs/logs/interface_v1_smoke.jsonl \
  --output outputs/submissions/interface_v1_smoke.xlsx
```

观察修复次数、兜底来源及最终答案，不只看成功计数。线上复测需已有模型密钥，
会产生 API 调用费用；本次实现未进行线上请求。

后续将证据选择质量、全图与局部图上下文作为独立实验，不混入本阶段评测。
