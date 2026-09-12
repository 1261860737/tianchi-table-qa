# 可控答案组装接口

目标：修复已读到数据之后的字段选择、集合粒度和百分数换算错误。不增加证据复核、
答案覆盖或新的模型重试。直接答案的保护边界保持不变。

## 实现位置

- `operation_contracts.py`：Pydantic 参数合同。
- `executor.py`：Decimal 运算、显式字段选择和集合计数。
- `pipeline.py`：已经选择返回字段的操作不再执行末尾投影，重放使用相同边界。
- `prompts.py`：专家及 Operation 修复共享同一份接口说明；format-only 的系统和用户
  指令复用同一常量，消除直接回答必须绑定 Evidence 的残留规则。
- `tests/test_answer_operations.py`：通用合成回归样本，不包含按评测题号处理的分支。

## 新参数（可选，兼容旧操作）

1. `list.select_field="entity"`：在引用解析前读取每条 Evidence 的指定字段。
   混合字段仍使用单项引用的 `field`。不根据问题关键词猜字段，不覆盖冲突的显式声明。
2. `argmax/argmin.return_field="label"`：直接返回标签；省略保持旧的完整记录返回。
   需要多个标签时先用 concat 构造完整实体。设置返回字段后不再二次投影。
3. `count.source={"evidence_index":0}`：统计一条数组证据中的元素。
   `values` 与 `source` 二选一；可用 `exclude_values` 声明精确排除项。
   不递归展开、不自动去重。已读到的数量使用 lookup，而不是 count([数量])。
4. 二元运算的 `a_unit/b_unit="percent"`：对应参数的百分数数值除以 100 再运算。
   默认 number，不根据裸数字推断单位；0.06 与 6% 必须区别声明。
   百分点差操作保持原含义。

这些参数约束执行含义，并不能替模型判断表格语义。旧日志缺失参数时不会自动补猜，
所以历史 success 结果不会仅因代码升级自动变正确；应先用新提示词做小规模模型回归。
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
