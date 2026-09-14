# 可控答案组装接口

目标：修复已读到数据之后的字段选择、集合粒度和百分数换算错误。不增加证据复核或
答案覆盖。答案、计算和结构分别使用最小必要接口，避免辅助协议字段否决有效答案。

## 三条执行通道

- Extract/Visual 纯抽取：`direct_answer` 是主数据，Evidence 仅用于审计。无效或退化 bbox
  按缺失处理；多字段答案直接按题目顺序输出数组，不为组装数组强制制造 Operation。
- Compute：必须提供最小 Evidence 与 Operation，最终值由 Python 执行器产生，模型不能
  使用 `direct_answer` 绕过计算。
- Structure：必须通过完整几何校验。校验器向修复器提供全部冲突，受限补丁可在有限轮次
  内只修改冲突单元格的坐标和跨度；候选整体合法后才成为答案。

## 实现位置

- `operation_contracts.py`：Pydantic 参数合同。
- `executor.py`：Decimal 运算、百分数单位识别、显式字段选择和集合计数。
- `pipeline.py`：在 Evidence 元数据丢失前绑定单字段列表投影；argmax/argmin 的标量
  返回形状由 answer_format 决定；已经选择返回字段的操作不再执行末尾投影。
- `prompts.py`：按三条执行通道约束专家职责；不向所有专家追加一整套新接口说明。
  format-only 的系统和用户指令复用同一常量。
- `tests/test_answer_operations.py`：通用合成回归样本，不包含按评测题号处理的分支。

## 新参数（可选，兼容旧操作）

1. `list.select_field="entity"`：在引用解析前读取每条 Evidence 的指定字段。
   混合字段仍使用单项引用的 `field`。不根据问题关键词猜字段，不覆盖冲突的显式声明。
2. `argmax/argmin.return_field="label"`：直接返回标签；省略保持旧的完整记录返回。
   需要多个标签时先用 concat 构造完整实体。设置返回字段后不再二次投影。
3. `count.values` 是唯一推荐接口。它可以包含多个逐项引用，也可以只引用一条数组
   Evidence；后一种由执行器剥一层后计数。可用 `exclude_values` 声明精确排除项。
   不递归展开、不自动去重。`source` 只为读取 v6 检查点保留，不再提供给模型。
   已读到的数量使用 lookup，而不是 count([数量])。
4. 二元运算的 `a_unit/b_unit="percent"`：对应参数的百分数数值除以 100 再运算。
   Evidence 的 `unit="%"` 或 `value_raw="8%"` 会由代码识别；裸数字不推断单位。
   显式指定 number 可表示已经归一化的 0.08。
   百分点差操作保持原含义。

这些参数约束执行含义，并不能替模型判断表格语义。历史日志中“单一 Evidence 引用解析
为数组”的 count 会按集合计数；其他缺失参数不会自动补猜。完整版本评估仍需从头运行
全部题目，小规模模型回归只用于在全量运行前发现协议级错误。
不建议自动拍平数组、将所有 argmax 强制返回 label，或把所有含百分号的读取值改成小数。

## 依赖与验证

沿用已有 Pydantic、pytest，数值计算使用标准库 Decimal，无新依赖。
首次安装：`pip install -e '.[dev]'`。

```bash
conda run -n tableqa python -m pytest -q
conda run -n tableqa ruff check src tests
```

本轮不修改结构坐标口径、路由选择、模型配置和提交文件。测试通过仅代表代码合同成立，
不代表模型选择字段一定正确或线上分数必然提高。

## 可控实验方式

`build-submission --replay-success-counts` 会沿用日志中的 Evidence、Operation 和
TaskPlan，用当前 Python 执行器重算安全的成功 count。它不调用模型，也不重新读图；因此
不会把 908 题的模型采样波动混入同一次实验。含空值、合计类成员且缺失显式排除规则的
旧集合会跳过。默认关闭，避免静默改写历史答案。
