# Agent.md

# 知乎知学堂大模型挑战赛：复杂表格识别与内容理解
## Document / Table QA Agent 总体方案

---

## 1. 项目目标

本项目面向「复杂表格识别与内容理解」任务，目标是构建一个可自动完成以下流程的文档智能 Agent：

> **输入：PDF / 表格图片 + 问题**  
> **输出：基于表格证据得到的最终答案**

系统不追求一开始就构建复杂 Multi-Agent，而是优先实现一个：

> **可运行、可解释、可观测、可迭代的 Table QA Baseline**

整体思想：

- 大模型负责：
  - 理解用户问题
  - 定位相关表格内容
  - 识别需要使用的数据
  - 判断问题类型
  - 选择计算工具
  - 输出结构化中间结果
- 程序负责：
  - 确定性数学计算
  - 单位换算
  - 数值格式化
  - 答案标准化
  - 日志记录
  - 自动评测

核心原则：

> **LLM 负责理解与决策，代码负责确定性执行。**

---

# 2. 不要一开始做“大而全”的 Agent

第一阶段不直接实现：

```text
Document Router
    ↓
OCR Agent
    ↓
Table Parser Agent
    ↓
Retriever Agent
    ↓
Planner Agent
    ↓
Reasoning Agent
    ↓
Calculator Agent
    ↓
Verifier Agent
    ↓
Answer Agent
```

原因：

1. Agent 链越长，错误传播越严重。
2. 很难判断错误到底来自哪个模块。
3. API 调用次数多，成本和延迟增加。
4. 当前还不知道真实数据集中最主要的错误类型。
5. 很多题实际上只需要找到 1~3 个 Cell。

因此第一阶段采用：

```text
Table Image / PDF Page
        +
     Question
        ↓
      Qwen
        ↓
Evidence + Operation JSON
        ↓
 Python Tool Executor
        ↓
     Final Answer
```

先建立最简单的 V0 Baseline。

---

# 3. 核心任务抽象

对于一道问题：

```text
2025 年华东地区营业收入相比 2024 年增长率是多少？
```

表格中：

| 地区 | 2024 营业收入 | 2025 营业收入 |
| --- | ---: | ---: |
| 华东 | 120 | 150 |
| 华南 | 100 | 110 |

系统真正需要完成的是：

```text
Question
   ↓
理解目标：
- entity = 华东
- metric = 营业收入
- time = 2024, 2025
   ↓
定位 Cell：
- 华东 × 2024营业收入 = 120
- 华东 × 2025营业收入 = 150
   ↓
判断 Operation：
percentage_change
   ↓
调用代码：
percentage_change(old=120, new=150)
   ↓
得到：
25%
```

因此，本项目核心不是单纯 OCR，而是：

> **Question-aware Cell Grounding + Structured Reasoning + Tool Execution**

---

# 4. 第一阶段：Baseline V0

## 4.1 输入

优先支持：

```text
{
    "question": "...",
    "image": "table_image / pdf_page"
}
```

如果真实比赛是一整份 PDF，则后续增加 Document Retrieval。

如果官方已经给出了目标表格，则直接进入 Table QA。

## 4.2 V0 Pipeline

```text
┌───────────────────────────────┐
│             Input             │
│                               │
│ Question + Table Image        │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Qwen Evidence Agent     │
│                               │
│ 1. 理解问题                   │
│ 2. 找相关 Cell                │
│ 3. 读取 value                 │
│ 4. 读取 unit                  │
│ 5. 判断 question_type         │
│ 6. 选择 operation             │
│ 7. 输出结构化 JSON            │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│        Python Executor        │
│                               │
│ lookup                        │
│ add                           │
│ subtract                      │
│ multiply                      │
│ divide                        │
│ ratio                         │
│ percentage_change             │
│ percentage_of_total           │
│ sum                           │
│ average                       │
│ max / min                     │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│      Answer Normalizer        │
└───────────────┬───────────────┘
                ↓
          Final Answer
```

---

# 5. 最关键的设计：Evidence JSON

不允许模型直接只输出：

```json
{
  "answer": "25%"
}
```

因为这样无法判断：

- 模型有没有找错 Cell
- 有没有看错年份
- 有没有单位错误
- 有没有算错
- 有没有 hallucination

模型必须输出中间 Evidence。

推荐格式：

```json
{
  "question_type": "percentage_change",
  "target": {
    "entity": "华东地区",
    "metric": "营业收入",
    "time_range": ["2024", "2025"]
  },
  "evidence": [
    {
      "row_header": "华东",
      "column_header": "2024营业收入",
      "value_raw": "120",
      "value": 120,
      "unit": "万元"
    },
    {
      "row_header": "华东",
      "column_header": "2025营业收入",
      "value_raw": "150",
      "value": 150,
      "unit": "万元"
    }
  ],
  "operation": {
    "name": "percentage_change",
    "arguments": {
      "old_value": 120,
      "new_value": 150
    }
  },
  "output": {
    "type": "percentage"
  }
}
```

---

# 6. 为什么 Evidence 必须包含 Row / Column / Unit

真正重要的不是模型识别出：

```text
120
150
```

而是：

```text
120
↑
华东
↑
2024
↑
营业收入
↑
万元
```

复杂表格中最容易出错的是：

- 多级表头
- 横向表头
- 纵向表头
- 合并单元格
- 跨行
- 跨列
- 单位写在表头外
- 财务报表中“本期 / 上期”
- 空单元格
- 同一数字在多个位置重复出现

因此我们真正需要优化的是：

> **Cell Grounding**

而不是单纯 OCR Accuracy。

---

# 7. Question Understanding

不要只做粗粒度的：

```text
查询
计算
比较
```

而是定义 Table QA 专用的 Operation Ontology。

第一版建议：

```text
lookup
直接查值

add
相加

subtract
相减

multiply
相乘

divide
相除

ratio
比值

percentage_change
增长率 / 同比 / 环比

percentage_of_total
占总量比例

sum
求和

average
平均

max
最大值

min
最小值
```

---

# 8. 为什么不只提供加减乘除

例如：

```text
同比增长率
```

如果只允许：

```text
subtract
divide
multiply
```

模型就必须规划：

```text
new - old
↓
÷ old
↓
× 100
```

增加多步规划错误。

更稳定的方式：

```text
percentage_change(old, new)
```

因此工具应该尽量使用：

> **语义级工具，而不是只有底层算术工具。**

---

# 9. Python Tool Executor

```python
def lookup(value):
    return value


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ZeroDivisionError
    return a / b


def ratio(a, b):
    if b == 0:
        raise ZeroDivisionError
    return a / b


def percentage_change(old_value, new_value):
    if old_value == 0:
        raise ZeroDivisionError
    return (new_value - old_value) / old_value * 100


def percentage_of_total(part, total):
    if total == 0:
        raise ZeroDivisionError
    return part / total * 100


def sum_values(values):
    return sum(values)


def average(values):
    return sum(values) / len(values)


def maximum(values):
    return max(values)


def minimum(values):
    return min(values)
```

后续根据比赛题型增加：

```text
同比
环比
复合增长率
单位转换
绝对值
排名
差额
倍数
区间
```

---

# 10. 第一阶段不做完整表格结构恢复

Baseline 不建议默认：

```text
Image
 ↓
完整 OCR
 ↓
恢复整张 Table JSON
 ↓
Search
 ↓
Answer
```

原因：

如果一个表格有：

```text
50 × 20 = 1000 cells
```

问题可能只需要：

```text
2 cells
```

完整恢复 1000 个 Cell：

- 成本高
- 错误机会多
- 对答案帮助有限

因此第一版优先尝试：

> **Query-aware Table Extraction**

即：

```text
Question + Table
        ↓
Qwen
        ↓
只寻找回答当前问题所需的最小证据
```

---

# 11. Prompt 核心约束

Qwen Evidence Agent 需要明确：

```text
1. 不要直接回答最终答案。

2. 先理解问题需要哪些信息。

3. 只从当前表格中寻找证据。

4. 记录每一个数据对应：
   - row_header
   - column_header
   - raw_value
   - value
   - unit

5. 不进行数学计算。

6. 如果需要计算，只输出 operation 和 arguments。

7. 所有结果严格输出 JSON。

8. 如果证据不足，不允许猜测。

9. 如果存在多个候选 Cell，需要输出候选或标记 ambiguous。

10. 不允许生成表格中不存在的数字。
```

---

# 12. V0 的 Question → Evidence 示例

## Case 1：直接查询

问题：

```text
2025 年华东营业收入是多少？
```

输出：

```json
{
  "question_type": "lookup",
  "evidence": [
    {
      "row_header": "华东",
      "column_header": "2025营业收入",
      "value_raw": "150",
      "value": 150,
      "unit": "万元"
    }
  ],
  "operation": {
    "name": "lookup",
    "arguments": {
      "value": 150
    }
  }
}
```

## Case 2：差值

问题：

```text
2025 年华东营业收入比 2024 年多多少？
```

输出：

```json
{
  "question_type": "difference",
  "evidence": [
    {
      "row_header": "华东",
      "column_header": "2024营业收入",
      "value": 120
    },
    {
      "row_header": "华东",
      "column_header": "2025营业收入",
      "value": 150
    }
  ],
  "operation": {
    "name": "subtract",
    "arguments": {
      "a": 150,
      "b": 120
    }
  }
}
```

## Case 3：同比增长

问题：

```text
2025 年华东营业收入同比增长多少？
```

输出：

```json
{
  "question_type": "percentage_change",
  "evidence": [
    {
      "row_header": "华东",
      "column_header": "2024营业收入",
      "value": 120
    },
    {
      "row_header": "华东",
      "column_header": "2025营业收入",
      "value": 150
    }
  ],
  "operation": {
    "name": "percentage_change",
    "arguments": {
      "old_value": 120,
      "new_value": 150
    }
  }
}
```

---

# 13. 必须记录内部运行日志

每一道题建议保存：

```json
{
  "question_id": "xxx",
  "question": "...",
  "document_id": "...",
  "page_id": "...",
  "evidence": [],
  "operation": {},
  "tool_result": "...",
  "final_answer": "...",
  "raw_model_output": "...",
  "latency": 2.31,
  "token_usage": 3421,
  "status": "success"
}
```

这样才能进行 Bad Case 分析。

---

# 14. Error Taxonomy

所有错误至少分类成：

```text
E1 Document Retrieval Error
找错文档

E2 Page Retrieval Error
找错页面

E3 Table Retrieval Error
找错表格

E4 Row Grounding Error
找错行

E5 Column Grounding Error
找错列

E6 OCR / Value Recognition Error
数字识别错误

E7 Unit Error
单位识别或换算错误

E8 Operation Selection Error
计算类型选择错误

E9 Argument Binding Error
数字选对，但 old/new 等参数顺序错误

E10 Tool Error
代码执行错误

E11 Output Normalization Error
格式错误

E12 Ambiguous / Insufficient Evidence
证据不足

E13 Hallucination
模型生成表格不存在的数据
```

---

# 15. 第一轮实验重点

V0 跑完以后，不要立即增加 Agent。

首先统计：

```text
总题数
准确数
错误数
每类 Error 占比
```

例如：

```text
100 个错误：

41  Cell Grounding
23  OCR / Number Recognition
14  Unit
10  Operation
7   Output Format
5   Other
```

这时候才能决定下一步优化方向。

---

# 16. V1：如果 Cell Grounding 是主要问题

加入：

```text
Question
   ↓
Query Analyzer
   ↓
Candidate Evidence Retriever
   ↓
Top-K Candidate Cells
   ↓
Evidence Selector
   ↓
Tool Executor
```

核心：

> 不直接让模型“一次猜对”，而是先召回候选，再选择。

中间格式：

```json
{
  "candidates": [
    {
      "row": "...",
      "column": "...",
      "value": "..."
    },
    {
      "row": "...",
      "column": "...",
      "value": "..."
    }
  ]
}
```

---

# 17. V2：如果大表格看不清

增加 Region / Crop Pipeline。

```text
Question
   +
Whole Table
      ↓
Coarse Grounding
      ↓
Region Proposal
      ↓
Crop Relevant Area
      ↓
High Resolution Qwen
      ↓
Evidence JSON
```

适用于：

- 超宽表格
- 超长表格
- 字体很小
- 多级表头
- 高分辨率财务报表

---

# 18. V3：如果输入是完整 PDF

增加 Document Retrieval。

```text
Question
   ↓
Question Analyzer
   ↓
Document / Page Retriever
   ↓
Relevant Pages
   ↓
Table Retriever
   ↓
Relevant Table
   ↓
Table QA Agent
```

需要解决：

```text
Document Grounding
Page Grounding
Table Grounding
Cell Grounding
```

这是一个逐级 Grounding 问题。

---

# 19. PDF Retrieval 可考虑的策略

第一版：

```text
PDF
 ↓
逐页转图片
 ↓
Qwen / OCR 获取粗粒度 page summary
 ↓
Question 与 Page Summary 匹配
 ↓
Top-K Pages
```

后续可以考虑：

```text
OCR Text Embedding
+
Question Embedding
+
关键词召回
+
LLM Re-rank
```

即：

```text
Retriever
 ↓
Top-K
 ↓
Qwen Reranker
 ↓
Target Page
```

---

# 20. V4：Verifier

只有当 V0/V1 已经较稳定后，再增加 Verifier。

Verifier 不重新完整做题。

它只检查：

```text
1. Evidence 是否真的支持问题？
2. Row 是否正确？
3. Column 是否正确？
4. Unit 是否一致？
5. Operation 是否匹配问题？
6. Argument 顺序是否正确？
```

结构：

```text
Question
   ↓
Evidence Agent
   ↓
Evidence JSON
   ↓
Verifier
   ↓
Pass / Reject
```

Reject 后：

```text
重新 Grounding
```

---

# 21. Verifier 不应直接看 Final Answer

推荐检查：

```text
Question
+
Evidence
+
Operation
```

而不是：

```text
Question
+
Final Answer
```

因为我们想验证的是推理链中的结构信息，而不是让另一个模型凭感觉判断答案。

---

# 22. V5：完整 Document Agent

最终可能演化为：

```text
                         Question
                            │
                            ▼
                    Question Analyzer
                            │
                 ┌──────────┴─────────┐
                 ↓                    ↓
             entities            operation hint
                 │
                 └──────────┬─────────┘
                            ↓
                   Document Retriever
                            │
                            ↓
                     Page Retriever
                            │
                            ↓
                    Table Retriever
                            │
                            ↓
                    Region Grounder
                            │
                            ↓
                   Candidate Evidence
                            │
                            ↓
                    Evidence Selector
                            │
                            ↓
                     Planner Agent
                            │
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
           Lookup       Calculator    Comparison
              │             │             │
              └─────────────┴─────────────┘
                            ↓
                      Verifier Agent
                            │
                            ↓
                     Answer Normalizer
                            │
                            ↓
                       Final Answer
```

注意：

> 这是最终方向，不是第一版开发目标。

---

# 23. 推荐工程目录

```text
table_qa_agent/
│
├── README.md
├── Agent.md
│
├── configs/
│   ├── model.yaml
│   ├── prompt.yaml
│   └── tools.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── samples/
│   └── submissions/
│
├── src/
│   ├── api/
│   │   └── qwen_client.py
│   ├── agents/
│   │   ├── evidence_agent.py
│   │   ├── planner_agent.py
│   │   ├── verifier_agent.py
│   │   └── query_analyzer.py
│   ├── retrieval/
│   │   ├── document_retriever.py
│   │   ├── page_retriever.py
│   │   ├── table_retriever.py
│   │   └── region_retriever.py
│   ├── tools/
│   │   ├── calculator.py
│   │   ├── unit_converter.py
│   │   └── normalizer.py
│   ├── schemas/
│   │   ├── evidence.py
│   │   ├── operation.py
│   │   └── result.py
│   ├── prompts/
│   │   ├── evidence_prompt.txt
│   │   ├── verifier_prompt.txt
│   │   └── query_prompt.txt
│   ├── pipeline/
│   │   ├── baseline.py
│   │   └── document_agent.py
│   ├── eval/
│   │   ├── evaluator.py
│   │   ├── error_analysis.py
│   │   └── metrics.py
│   └── utils/
│       ├── logger.py
│       ├── json_utils.py
│       └── image_utils.py
│
├── scripts/
│   ├── run_baseline.py
│   ├── run_eval.py
│   ├── analyze_errors.py
│   └── generate_submission.py
│
├── logs/
│
└── outputs/
    ├── predictions/
    ├── evidence/
    └── reports/
```

---

# 24. Baseline 第一版最小工程

第一版实际上只需要：

```text
qwen_client.py
evidence_agent.py
calculator.py
normalizer.py
baseline.py
run_baseline.py
evaluator.py
```

不要一开始实现整个目录。

---

# 25. 第一阶段开发顺序

## Step 1：数据分析

先统计真实比赛数据：

```text
PDF 数量
Image 数量
Question 数量
表格类型
平均页数
表格大小
问题类型
```

重点统计：

```text
直接查询
差值
求和
比例
增长率
平均
最大/最小
跨行
跨列
跨页
跨表
```

## Step 2：Qwen Direct Baseline

先测试：

```text
Question + Image
↓
Qwen
↓
直接 Answer
```

目的不是最终采用。

而是得到：

> 最基础的性能参考。

记录 Accuracy。

## Step 3：Structured Baseline

实现：

```text
Question + Image
↓
Evidence JSON
↓
Tool Executor
↓
Answer
```

然后和 Direct Baseline 对比。

## Step 4：Error Analysis

统计：

```text
Direct Answer Error
vs
Structured Agent Error
```

重点判断：

```text
Structured Evidence 是否真的提升了：
- 稳定性
- Grounding
- 数学准确率
- 可解释性
```

## Step 5：针对最大错误源优化

原则：

> 每次只解决最大的一个问题。

例如：

```text
Grounding 错最多
→ 做 Candidate Retrieval

数字看错最多
→ 做 Crop / OCR

单位错最多
→ Unit Module

计算类型错最多
→ Operation Ontology

大表问题最多
→ Region Grounding

跨页问题最多
→ Page Retriever
```

---

# 26. 实验路线

建议至少保留以下实验：

```text
Exp-0
Qwen Direct Answer

Exp-1
Qwen Evidence + Python Calculator

Exp-2
Question Analyzer + Evidence

Exp-3
Candidate Retrieval + Evidence Selector

Exp-4
Crop + High-resolution Evidence

Exp-5
Verifier

Exp-6
Full Document Retrieval
```

---

# 27. Ablation

后续可以做：

```text
w/o Evidence JSON

w/o Calculator

w/o Unit Normalization

w/o Candidate Retrieval

w/o Crop

w/o Verifier

Single-pass vs Multi-stage

Direct Answer vs Tool Execution
```

---

# 28. 评测维度

除了最终 Accuracy，还应该记录：

```text
Answer Accuracy
Evidence Accuracy
Operation Accuracy
Value Recognition Accuracy
Unit Accuracy
Grounding Accuracy
JSON Parse Success Rate
Tool Execution Success Rate
Average API Calls
Average Tokens
Average Latency
Average Cost
```

---

# 29. 最重要的 Debug 指标

如果 Answer 错，必须能够回答：

```text
模型是否找到了正确的 Cell？

如果没有：
Grounding 问题。

如果找到了：
Value 是否读对？

如果读对：
Operation 是否正确？

如果 Operation 正确：
Argument 是否绑定正确？

如果都正确：
Tool / Normalizer 是否出错？
```

这样才能让系统真正可迭代。

---

# 30. 一个重要设计原则：不要让 LLM 偷偷计算

Prompt 中明确：

```text
不要计算最终结果。
```

模型只允许：

```text
找到：
120
150

选择：
percentage_change
```

Python 得：

```text
25%
```

原因：

- 防止算术 hallucination
- 方便 Debug
- 方便复现
- 方便验证
- 方便统计 Operation Accuracy

---

# 31. 一个重要设计原则：不要过早强制完整 OCR

我们的默认路线：

```text
Question-aware Extraction
```

而不是：

```text
Full-table Parsing First
```

但是如果后续实验发现完整 Table Representation 能够显著提升 Grounding，则可以增加：

```text
Image
↓
Table Parser
↓
Structured Table
↓
Question QA
```

这应该由实验决定，而不是提前假设。

---

# 32. 一个重要设计原则：逐级 Grounding

如果最终任务是：

```text
完整 PDF + Question
```

则整个问题本质上是：

```text
Document Grounding
        ↓
Page Grounding
        ↓
Table Grounding
        ↓
Region Grounding
        ↓
Cell Grounding
        ↓
Operation Grounding
```

每一层都可以单独评测。

这可能成为整个系统最核心的设计思想。

---

# 33. 当前阶段最值得研究的方向

按照优先级：

## P0：Query-aware Cell Grounding

重点研究：

```text
Question
↓
哪些 Row / Column / Cell 真正相关？
```

这是第一核心问题。

## P1：复杂表头理解

重点：

```text
多级表头
合并单元格
横纵层级关系
年份 / 指标 / 地区关系
```

## P2：Evidence + Tool Reasoning

重点：

```text
Evidence
↓
Operation
↓
Arguments
```

减少 LLM 自由推理。

## P3：Large Table Region Grounding

重点：

```text
Whole Table
↓
Coarse Locate
↓
Crop
↓
Fine Recognition
```

## P4：Document Retrieval

如果赛题是完整 PDF：

```text
Question → Page → Table → Cell
```

## P5：Verifier

只在基础系统稳定后加入。

---

# 34. 暂时不优先研究

第一阶段不优先：

```text
复杂 Multi-Agent 协同
大规模 Fine-tuning
RL
训练自己的 OCR
训练 Table Structure Recognition 模型
复杂 RAG 框架
长 Chain-of-Thought
模型自动写 Python
自由代码执行
```

除非 Bad Case 证明它们是主要瓶颈。

---

# 35. 最终目标

最终系统应该达到：

```text
输入：
PDF / Image
+
Question

↓
自动定位文档证据

↓
输出可解释 Evidence

↓
产生结构化 Operation

↓
使用确定性 Tool

↓
Verifier 检查

↓
输出比赛要求的最终答案
```

并且对于每个错误，都可以回答：

> **它到底错在感知、Grounding、Planner、Tool，还是格式化？**

---

# 36. 当前执行策略总结

现阶段坚持：

```text
Baseline First
↓
Collect Errors
↓
Classify Errors
↓
Find Largest Bottleneck
↓
Only Add Necessary Module
↓
Re-evaluate
```

不要：

```text
先设计一个看起来很完整的 Agent
↓
全部模块一起实现
↓
最后不知道为什么错
```

---

# 37. 当前 V0 最终定义

第一版只实现：

```text
Question
+
Table Image
↓
Qwen Evidence Agent
↓
Structured JSON
↓
Python Tool Executor
↓
Answer Normalizer
↓
Final Answer
```

第一阶段目标不是追求最强成绩。

第一阶段目标是：

> **把一个可解释、可复现、可自动评测的完整 Pipeline 跑通。**

然后基于真实数据和真实 Bad Case，决定系统下一步向：

```text
Retriever
Crop
OCR
Table Parsing
Verifier
Multi-Agent
```

中的哪一个方向演化。

---

# 38. 一句话总结

本项目的核心不是：

> “让 Qwen 看表格然后回答问题。”

而是：

> **把复杂表格问答拆成 Grounding → Evidence → Operation → Tool Execution → Verification 的结构化文档智能 Agent。**

第一阶段从最小可运行 Baseline 开始，通过 Error-driven Iteration 逐步演化为完整 Document Agent。
