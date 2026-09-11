# 数据检查记录

检查对象：项目根目录的 `tests.xlsx`、`submit-template.xlsx` 和 `files/`。

## 工作簿结构

`tests.xlsx` 共 908 行，字段为：

```text
id, file_name, question_type, question, table_hint, answer_format, answer
```

`answer` 全为空，说明当前文件是待预测测试集，不是带标签的训练/验证集。`submit-template.xlsx` 只有 `id, answer` 两列表头。

## 题型和输出

| question_type | 数量 |
| --- | ---: |
| extract（原始合法行） | 617 |
| thinking | 204 |
| structure | 86 |
| 错位行 | 1 |

错位行为 ID 63，`question_type` 与 `question` 都是“关键管理人员薪酬本期发生额和上期发生额分别是多少？”。结合 `answer_format=json_array` 和相邻题目，它按 `extract` 修复。修复后的分布为 `extract=618`、`thinking=204`、`structure=86`。

| answer_format | 数量 |
| --- | ---: |
| string | 409 |
| number | 360 |
| json_array | 99 |
| json | 40 |

ID 32 的题目是“恢复前 2 行和第一列的表格结构”，但原始 `answer_format=number`，
与任务内容及同类 structure 题冲突，因此按 `json` 修复。清洗后的分布为
`string=409`、`number=359`、`json_array=99`、`json=41`。

因此 V0 不能只实现数值运算，还必须支持原文抽取、列表和表格结构 JSON。

## 文档情况

`files/` 中有 26 个 PDF 和 68 张图片。PDF 页数分布：18 个单页、6 个两页、1 个三页、1 个八页。题目实际引用 90 个文档，其中引用到的 PDF 最长为 3 页；8 页 PDF 未被当前测试表引用。目录中的部分文件未被当前测试表引用。

三个题目文件名与实际文件名不一致：

| 题目表 | 实际文件 |
| --- | --- |
| 58.pdf | 058.pdf |
| 59.pdf | 059.pdf |
| 0060.pdf | 060.pdf |

数据层采用“先精确匹配，再以数字 stem + 扩展名唯一匹配”的通用规则，不把三个名称写死。

## 对 V0 的影响

1. PDF 必须保留页序并支持至少 8 页。
2. `table_hint` 有 53 个空值，不能依赖它完成定位。
3. `structure` 同时可能要求 JSON 结构或行/列计数。
4. `thinking` 不只是四则运算，还包含计数、列表组成、最大/最小和真假判断。
5. 测试集无真值，首轮只能验证结构可运行性；准确率评估需要官方结果或另行构造小型标注集。
