# 提交答案协议

程序把比赛提交规范固化为运行时约束，而不是只依赖模型提示词。

## 工作簿

- 输出文件为 `.xlsx`，默认只有 `id`、`answer` 两列。
- `id` 必须与题目清单一致且不能重复。
- 无法作答时禁止写 `None`、`null` 或实际空白单元格。为兼容评测器的非空与类型校验，程序按格式写显式占位值：`string/number` 写 `""`，`json_array` 写 `[]`，结构恢复 `json` 写 `{"row_count":0,"col_count":0,"cells":[]}`。

## 结构恢复

`answer_format=json` 的题目必须输出：

```json
{
  "row_count": 4,
  "col_count": 3,
  "cells": [
    {
      "text": "项目",
      "row": 0,
      "col": 0,
      "rowspan": 2,
      "colspan": 1
    }
  ]
}
```

- `row_count`、`col_count` 是完整表格的逻辑行列数。
- 局部恢复时，`cells` 只包含题目要求范围，坐标仍相对完整表格从 0 开始。
- 合并单元格只输出左上角一次；普通单元格跨度均为 1。
- 单元格不得越界、重叠或包含协议外字段。

## 其他答案

- `number`：必须能解析成数字，不能包含千分位逗号。
- `json_array`：必须是合法 JSON 数组，内部空值使用 `""`，禁止 `null`。
- `string`：保留原始含义；禁止添加“根据表格可知”“答案是”等前缀。

`table-qa run` 会在单题归一化后检查协议。`table-qa validate-submission` 会再次逐题
检查最终工作簿；发现格式错误时返回非零退出码。空答案符合提交格式，但会单独统计。

仅查看工作簿无法判断空值是“原图中确实没有目标字段”还是“运行异常”。程序提供两种
答案策略：

- 默认 `format-only`：Evidence 只用于日志和审计，不作为拒答条件；模型必须给出最佳
  判断，残余缺失使用类型安全的非空兜底。
- 可选 `strict-evidence`：`ambiguous/insufficient` 写入显式空占位；`error` 且没有合法
  历史成功答案时拒绝发布。

两种策略都会保留历史回退和 Evidence 重放能力，也都会执行最终答案格式校验。

## 原子发布

`table-qa run` 和 `table-qa build-submission` 不会直接覆盖正式提交文件。程序先写入同目录
临时候选文件，完成工作簿协议校验后，再以原子替换方式发布到 `--output`。严格证据模式
还要求不存在“运行错误导致的空答案”。如果生成或校验失败，旧文件保持不变，临时候选
文件会自动清理，命令返回非零退出码。

## 证据校验边界

默认 `format-only` 策略不因 Evidence 绑定或证据不足拒绝答案，并为剩余失败写入类型安全
的非空兜底。它只放宽证据层门禁；ID 完整性、重复 ID、工作簿列和最终答案格式仍必须
通过校验。使用 `--answer-policy strict-evidence` 可恢复旧版严格行为。

在写固定兜底前，流水线会尝试本地恢复可见内容。字符串依次考虑 `entity`、列头、行头、
归一化值、原始值和原文；数字和数组使用对应的 Evidence 值。Evidence 引用允许
`value`、`value_raw`、`row_header`、`column_header`、`unit`、`entity`、`metric` 和
`source_text`，按 `evidence_id` 与 `evidence_index` 引用时规则一致。
