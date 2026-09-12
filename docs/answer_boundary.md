# 答案与辅助证据边界

本次变更在接口简化基础上继续实现，不代表实验版源码恢复。

- Extract/Visual 的非结构 JSON 直接答案，先按题目输出格式检查。
  合法答案不因 Evidence 缺失、重复 ID、错误 bbox、辅助计划或操作不合法而被否决。
- `mode` 不再决定是否接受直接答案；答案类型由题目决定。
- 直接答案不再经模型额外指定的投影、精度配置改写。
- 原始响应继续存入日志；辅助信息无法解析时记录警告，不重新请求模型。
- Compute 仍需 Operation 执行；Structure 仍需真实行列结构满足输出合同。
  这两类执行失败不再通过取第一条证据或未知结构标记运行成功。
- 提交层仍保留原有 format-only 非空兜底，失败状态不会因此改为正常成功。
- 本轮没有取消全部恢复机制。无合法直接答案的抽取仍沿用原执行/恢复路径；
  严格证据模式仍检查计算参数来源。后续可分别评估，不能把本次当作无条件放行所有输出。

依赖保持不变：`pip install -e '.[dev]'`。

```bash
pytest -q
ruff check src tests
table-qa run --ids 51,84,126,146,260,698,759,838,859 --no-resume --workers 4 \
  --log-path outputs/logs/answer_boundary_smoke.jsonl \
  --output outputs/submissions/answer_boundary_smoke.xlsx
```

此命令只对照关键样本；84 的证据计数粒度和结构题识别质量不是本次修复目标。
先检查已有答案是否保留，再决定是否全量评测。取消协议否决不保证模型内容正确。
