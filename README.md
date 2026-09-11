# Table QA Agent V1

面向「复杂表格识别与内容理解」的可解释基线。系统把一道题拆成：

```text
Question + PDF/Image
        -> Function Call Intent Planner
        -> Python Plan Policy / Router（失败时规则 Planner 兜底）
        -> Extract / Compute / Structure / Visual Agent
        -> direct_answer 或受约束的原生 tool_call
        -> 共享 Locator / 高清 ROI / OCR 能力
        -> Evidence + Python Operation/DAG Executor（计算题）
        -> Evidence 软检查（只记录风险，不修复、不覆盖）
        -> Answer Projection
        -> Answer Normalizer
        -> Submission + JSONL Trace
```

系统使用轻量级分层专家架构。模型 Intent Planner 通过强制 Function Call 只声明题型、
所需字段、操作和能力，Python 再校验计划并选择路径。Extract/Visual 的 scalar 单值题允许
绑定 Evidence 后直接回答，Compute/Structure/multi-field 必须把操作交给 Python；
Extract、Compute 和 Structure 可以通过原生 Function Calling 主动请求
`inspect_table_region`。工具底层复用 Locator → 高清 ROI → OCR，不在各 Agent 中重复实现。
Evidence 检查只记录风险，不再触发第二候选或覆盖原答案；硬协议失败仍可进入有限恢复。

## 数据结论

对项目根目录 `tests.xlsx` 和 `files/` 的实际检查结果：

| 项目 | 数量 |
| --- | ---: |
| 题目 | 908 |
| 题目涉及的文档 | 90 |
| extract（清洗后） | 618 |
| thinking | 204 |
| structure | 86 |
| PDF | 26（目录内最长 8 页；测试题引用最长 3 页） |
| 图片 | 68 |

原始输出格式为 `string` 409 题、`number` 360 题、`json_array` 99 题、`json` 40 题。数据清洗后为 `string` 409 题、`number` 359 题、`json_array` 99 题、`json` 41 题。原始题目表还包含以下已知问题：

- `58.pdf` 实际文件为 `058.pdf`；
- `59.pdf` 实际文件为 `059.pdf`；
- `0060.pdf` 实际文件为 `060.pdf`；
- ID 63 的 `question_type` 被误填为问题正文，实际按 `extract` 处理。
- ID 32 要求恢复表格结构，但 `answer_format` 被误标为 `number`，实际按 `json` 处理。

这些规则已经固化在数据层，运行时会记录实际解析到的文件名。完整说明见 [docs/data_findings.md](docs/data_findings.md)，方法与模型选择见 [docs/design_decisions.md](docs/design_decisions.md)。

## 技术选型

- `openai`：连接阿里云百炼或其他 OpenAI 兼容视觉模型；
- `pydantic`：约束 Evidence、Operation 和日志协议；
- `PyMuPDF + Pillow`：PDF 逐页转图、图片方向修正和压缩；
- `pandas + openpyxl`：读取题目工作簿并生成 `id/answer` 提交；
- `json-repair`：修复模型偶发的轻微 JSON 语法错误，再交给 Pydantic 严格验证；
- `Typer + Rich + Loguru + tqdm`：CLI、日志和进度；
- `pytest + ruff`：离线测试和静态检查。

## 项目结构

```text
.
├── configs/baseline.yaml        # 模型、转图、运行配置
├── docs/data_findings.md        # 数据检查记录
├── docs/submission_contract.md  # 比赛提交协议与硬校验规则
├── src/table_qa_agent/
│   ├── agent.py                 # Evidence Agent 与协议校验
│   ├── agents/                  # 四类专家 Agent 注册
│   ├── capabilities/            # Agent 共享工具定义与参数校验
│   ├── answer/                  # 完整工具结果的字段投影
│   ├── client.py                # OpenAI 兼容视觉模型客户端
│   ├── config.py                # YAML 配置与环境变量覆盖
│   ├── dataset.py               # Excel 读取、清洗、文件名解析
│   ├── documents.py             # PDF/图片预处理和缓存
│   ├── executor.py              # 白名单单步/DAG 确定性操作
│   ├── normalizer.py            # 四类答案格式标准化
│   ├── ocr/                     # 可插拔 OCR 接口与视觉 OCR 后端
│   ├── orchestration/           # Task Planner 与确定性 Router
│   ├── pipeline.py              # 批处理、日志、断点续跑、提交
│   ├── prompts.py               # Evidence Prompt
│   ├── retrieval/               # 失败后的页面区域定位
│   ├── structure/               # 结构答案安全几何修复
│   ├── verification/            # 不拒答的 Evidence 软风险检查
│   └── schemas.py               # Pydantic 数据协议
├── tests/                       # 不调用 API 的测试
└── pyproject.toml
```

## 安装

推荐使用独立 Conda 环境：

```bash
cd /Users/cyh/Desktop/tianchi
conda create -n tableqa python=3.12 -y
conda activate tableqa
python -m pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

依赖以 `pyproject.toml` 为准。当前环境已使用可编辑安装，修改 `src/` 后无需重复安装。

## 配置

不要把 API Key 写进仓库：

```bash
export DASHSCOPE_API_KEY="sk-..."
export TABLE_QA_BASE_URL="https://你的工作空间地址/compatible-mode/v1"
export TABLE_QA_MODEL="qwen3.7-plus"
```

也可以在项目根目录创建 `.env`（程序会自动加载），或直接修改 `configs/baseline.yaml` 中除 API Key 外的参数。Base URL 优先读取 `TABLE_QA_BASE_URL`，同时兼容 `BASE_URL`。百炼不同地域/工作空间的 Base URL 不同，应以控制台为准。

## 运行

先检查数据，不调用模型：

```bash
table-qa inspect-data --input tests.xlsx --files-dir files
```

检查 PDF 转图和文件名纠错，不调用模型：

```bash
table-qa preprocess --ids 1,63,908
```

检查专家路由和 TaskPlan，不调用模型：

```bash
table-qa inspect-plan --ids 66,74,94,569
```

先跑 3 题冒烟：

```bash
table-qa run --limit 3 \
  --output outputs/submissions/smoke.xlsx \
  --log-path outputs/logs/smoke.jsonl
```

确认效果后运行完整测试集：

```bash
table-qa run \
  --output outputs/submissions/submission_v0.xlsx \
  --log-path outputs/logs/v0.jsonl \
  --workers 8
```

相同日志路径默认启用断点续跑，已经成功的题不会重复请求：

```bash
table-qa run \
  --output outputs/submissions/submission_v0.xlsx \
  --log-path outputs/logs/v0.jsonl \
  --workers 8 \
  --resume
```

配置默认也是 8 并发。若接口触发限流，可临时改成 `--workers 4`，或在
`configs/baseline.yaml` 中设置 `request_interval_seconds` 进行全局节流。
`planning.model_intent_enabled` 默认开启，因此每题会先增加一次无图片的意图规划调用；
规划输出不合法或服务端未返回 Function Call 时，自动使用本地规则 Planner。做旧流程消融时
可将其设为 `false`。
`recovery.max_agent_tool_calls` 控制专家的主动工具预算，默认 1；设为 0 可关闭专家阶段的
`inspect_table_region` Function Calling，同时保留 Intent Planner 和普通失败恢复。

新一轮运行时可以重复指定旧日志作为回退来源。程序只在当前结果不可用时采用较旧的
合法成功答案，避免协议偶发失败把上一版非空答案覆盖成空值：

```bash
table-qa run \
  --output outputs/submissions/submission_v2.xlsx \
  --log-path outputs/logs/v2.jsonl \
  --fallback-log outputs/logs/v1.jsonl \
  --fallback-log outputs/logs/full_v0.jsonl
```

也可以不调用模型，直接用当前代码重放旧日志中的 Evidence，并合并历史成功答案：

```bash
table-qa build-submission \
  --log-path outputs/logs/v1.jsonl \
  --fallback-log outputs/logs/full_v0.jsonl \
  --override-file configs/submission_overrides.yaml \
  --output outputs/submissions/submission_v2.xlsx
```

`configs/submission_overrides.yaml` 只保存人工查看原始文档后确认、且无法安全自动修复的
少量答案；它不是直接修改工作簿。每次执行命令都会从日志和该 YAML 重新生成完整文件，
并按题目 `answer_format` 校验每条覆盖值。默认 `format-only` 策略会自动补齐剩余答案，
因此完全自动生成时可省略 `--override-file`。

首次识别失败时，默认最多进行一次视觉恢复：先定位最多 3 个候选区域，再从原始
PDF 以 360 DPI 重绘或从原始图片裁剪，生成 OCR 候选后交给对应专家重试。可在
`retrieval`、`ocr`、`recovery` 配置段关闭或调整。

验证提交文件：

```bash
table-qa validate-submission outputs/submissions/submission_v0.xlsx --input tests.xlsx
```

校验器会逐题检查数字、JSON 数组和结构恢复协议。结构恢复题必须包含完整的
`row_count`、`col_count`、`cells`，并验证坐标、跨度、越界和重叠。发现格式错误时
命令返回非零退出码；空答案按比赛规则允许，但会列出对应 ID。详细协议见
[docs/submission_contract.md](docs/submission_contract.md)。

默认 `format-only` 策略仍记录 Evidence、状态和错误供审计，但不会用它们拒绝答案或
重复拦截模型结果。若模型未生成合法非空答案，程序写入类型安全的非空兜底。
只有显式指定 `--answer-policy strict-evidence` 时，证据不足可写空占位，运行错误造成的
空答案会触发发布门禁。

两个生成命令都采用原子发布：先在输出目录生成临时候选文件，再校验扩展名、列、ID、
答案类型；严格模式还检查错误空值。全部通过后才替换 `--output` 指定的正式文件；任一检查失败时返回
非零退出码、删除临时候选文件，并保留原有提交文件。因此可以把同一命令安全地重复运行。

### 答案策略

默认策略是 `format-only`。从头运行时无需额外实验参数：

```bash
table-qa run \
  --input tests.xlsx \
  --files-dir files \
  --config configs/baseline.yaml \
  --answer-policy format-only \
  --no-resume \
  --workers 8 \
  --log-path outputs/logs/format_only_v0.jsonl \
  --output outputs/submissions/submission_format_only_v0.xlsx
```

`--answer-policy format-only` 可省略，因为它已是默认值。该策略要求模型不得返回
`ambiguous/insufficient`，并跳过 Evidence 与 operation 的绑定
检查，允许模型直接给出最佳判断。最终答案的 number/json_array/json 协议校验仍会执行。
若网络、解析或模型输出异常仍留下答案缺失，程序按格式写入非空兜底：字符串为“未知”、
数字为 `0`、数组为 `["未知"]`、结构为合法的一格“未知”表。兜底只保证非空和格式正确，
不代表答案有视觉证据支持。Evidence 仍完整保存在 JSONL 日志中，可用于后续误差分析。

运行阶段还有一层本地恢复：在 `format-only` 下，Operation、投影、结构修复或答案归一化
失败时，优先从已提取 Evidence 的 `entity`、表头和值中恢复符合题目格式的答案，并把恢复
原因写入逐题 `warnings`；只有无法恢复时才使用上述固定兜底。极值问题由确定性 TaskPlan
根据问法和 `answer_format` 选择 `label` 或 `value`，避免把完整 `{label, value}` 记录写入
字符串答案。

中断后可用同一日志继续运行（去掉 `--no-resume`），或只重建提交：

```bash
table-qa build-submission \
  --input tests.xlsx \
  --log-path outputs/logs/format_only_v0.jsonl \
  --answer-policy format-only \
  --output outputs/submissions/submission_format_only_v0.xlsx
```

如需复现旧版硬证据门禁，用 `--answer-policy strict-evidence`。旧参数
`--experimental-all-answers` 仍然兼容，等价于 `--answer-policy format-only`。

当协议升级后继续使用旧 JSONL 日志，`--resume` 会重新执行旧日志中“状态成功但答案
不符合当前协议”的题目，不会错误跳过它们。

## 测试

```bash
pytest -q
ruff check .
```

测试覆盖数据清洗、文件名纠错、PDF/图片预处理、计算器、Evidence 字段引用、答案投影、
空答案恢复、答案格式和提交结构，不会调用外部模型。

## 运行产物

- `outputs/cache/documents/`：按输入文件指纹缓存的 JPEG 页面；
- `outputs/logs/*.jsonl`：每题 Evidence、Operation、工具结果、原始模型输出、Token、耗时和错误；
- `outputs/submissions/*.xlsx`：最终 `id/answer` 两列表。

## V1 边界与下一步

当前版本已经完成任务路由、专家 Prompt、类型化 Evidence、通用 Operation、答案投影、
区域定位、高清裁剪和 OCR 候选恢复。Verifier 尚未实现；应先用真实分数和小型人工验证集
评估 V1，再决定是否只对高风险题增加条件式验证。

接口实现参考阿里云百炼的 [OpenAI 兼容 Chat 文档](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions) 与 [视觉理解模型说明](https://help.aliyun.com/zh/model-studio/vision-model/)。
