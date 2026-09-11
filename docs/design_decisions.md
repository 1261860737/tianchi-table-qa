# V1 设计决策

## 从单 Agent 升级为受代码约束的分层专家

V1 不采用多个 Agent 自由讨论或投票。模型 Intent Planner 先通过强制 Function Call 声明
任务类型、原始字段、候选 Python Operation 和视觉能力，Python Policy 再校验硬元数据、
任务与专家匹配关系及操作白名单，并在规划失败时使用规则 Planner。通过校验后才进入
Extract、Compute、Structure、Visual Attribute 四类工作流。它们共享文档处理、区域定位、
OCR、数值解析和 Python Executor，但输出合同不同：Extract/Visual scalar 可在绑定 Evidence
后直接回答，Compute/Structure/multi-field 必须使用 Operation。生产逻辑按能力路由，不包含
题目 ID 分支。

模型输出保留完整关联记录，Python 操作可以返回包含 label/value 的完整对象，再由
Answer Projection 按题目要求选择字段。例如 argmax 返回最大记录，问名称时投影 label，
问数值时投影 value，多字段问题按声明顺序输出。

## 通用失败恢复

恢复策略按失败层级触发：Evidence 正确但 Operation 错误时只修复 Operation；证据不足、
空值或视觉协议失败时，Locator 先选择页内候选区域，再从原始 PDF 高清重绘或从原始图片
裁剪，并可生成 OCR 候选后交给原专家重试。结构答案只自动修复可由几何确定的维度偏小和
完全重复单元格；语义重叠不会被静默删除。

## 为什么采用 Question-aware Evidence + Python Tool

测试题通常只需要少量单元格，但文档包含多级表头、合并单元格、中英文混排、财务单位和多页 PDF。V0 直接把问题与相关文档页面交给视觉模型，让模型只返回最小 Evidence，再由 Python 执行确定性操作。

这一方案的主要收益：

1. 不必为回答两个单元格的问题恢复整张大表，减少无关 OCR 和结构恢复错误。
2. Evidence 保留行头、列头、原始值、单位和页码，能够区分 Cell Grounding、OCR、Unit 和 Operation 错误。
3. 模型不执行算术，避免简单计算错误；程序只开放白名单操作，不运行模型生成的代码。
4. 意图规划不读取图片；可通过配置关闭，方便对额外调用的收益、成本和延迟做消融。
5. JSONL 保留原始响应与中间结果，后续可以用真实 Bad Case 决定是否增加 Crop、Retriever 或 Verifier。

模型输出的计算参数使用 `evidence_index` 引用原始证据。复合问题输出一个小型 DAG，
后序步骤用 `step_id` 引用前序结果。例如“两个坏账准备之和占两个余额之和的比例”会被
表示成 `add → add → percentage_of_total`。参数校验会拒绝既不在 Evidence 中、也不是
前序工具结果的数字，因此“模型先心算，再让 Python 复述答案”无法通过协议。

## 为什么是单 Agent Baseline

这里的 Agent 不是让大模型包办答案，而是一个受限的规划器：负责看图、定位单元格、
填写 Evidence 引用并选择白名单操作。文件解析、算术、格式化、错误分类、并发、断点续跑
和提交生成都由确定性代码完成。这比纯 OCR 全表解析更贴近题目驱动的问答，也比自由式
端到端回答更容易审计。

批处理默认使用 8 个线程并发单题请求。线程数只提高吞吐，不改变单题协议；共享文档缓存、
JSONL 写入和请求节流均已做线程安全处理。接口限流时可降低 worker 或设置全局请求间隔。

## 为什么不先做完整 OCR / Table Parser

完整解析对需要全表导出的任务很合适，但当前目标是问答。对大表进行全量 OCR 和结构恢复会引入更多 Cell、合并关系和阅读顺序错误，并增加实现与调试变量。在没有标注答案的情况下，同时引入 OCR、Table Parser、Retriever 和 QA，很难定位收益来自哪个模块。

如果首轮日志显示主要错误确实是数字识别或复杂表头理解，再增加专用 OCR 或表结构解析器更合理。

## 为什么 Evidence 验证只记录风险

当前 908 题没有公开真值标签，无法证明第二个候选比原答案更可靠。硬 Evidence 门禁曾把
可用候选变成空答案，软恢复也曾因“协议可执行”而错误覆盖正确答案。因此验证器只输出
高置信度风险（缺 Evidence、Operation 未绑定等）并保留原非空答案，不再因软风险生成或
采用第二候选。只有协议解析、执行或答案格式等硬失败才进入有限恢复。

## 为什么默认 qwen3.7-plus

默认模型需要同时满足视觉输入、多图片、中文表格理解、长上下文和结构化 JSON。`qwen3.7-plus` 在百炼 OpenAI 兼容接口中覆盖这些能力，并且相较 `qwen3.8-max` 更适合先跑 908 题的成本可控基线。

配置中关闭思考模式，原因是本系统不希望模型自由计算；模型只负责 Evidence 和 Operation，且结构化输出更需要稳定、简短的 JSON，而不是长推理文本。

真实冒烟中，3 道顺序题平均约 8.41 秒/题；8 道并发回归约 19.8 秒完成，折合约
2.47 秒/题，吞吐提升约 3.4 倍。该数字是当前网络与样本下的观察值，不代表服务端 SLA。

模型名没有写死在代码里，可通过 `TABLE_QA_MODEL` 或 YAML 替换：

- `qwen3.8-max`：在 V0 流程稳定后做准确率优先实验；
- `qwen3.8-flash`：做低成本/低延迟对照；
- `qwen3.5-ocr`：当 Bad Case 证明主要瓶颈是 OCR，而不是 Cell Grounding 或推理时，作为前置识别模块；
- 本地开源 Qwen-VL：有稳定 GPU、需要离线或需要控制长期调用成本时再评估。

官方资料：

- https://help.aliyun.com/zh/model-studio/vision-model/
- https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions
- https://help.aliyun.com/zh/model-studio/qwen-structured-output
- https://help.aliyun.com/zh/model-studio/model-pricing

## 为什么把 PDF 转成图片

当前项目需要兼容 JPEG、PNG、WEBP 和 PDF，也希望保留清晰的页码 Evidence。统一转成 JPEG 页面后，所有文件走同一套视觉输入协议，缓存可复用，并且未来容易在页面级增加 Crop 或 Retrieval。

直接传 PDF 可能减少本地预处理，但会绑定特定模型和接口，页级图像控制也更弱。V0 先选择跨 OpenAI 兼容视觉模型更通用的多图 Data URL 方案。

## 何时调整路线

完整运行后按以下顺序决策：

1. Cell Grounding 错误最多：增加候选 Cell/区域召回。
2. OCR 错误最多：提高局部区域分辨率或引入 qwen3.5-ocr/PaddleOCR。
3. 多页定位错误最多：增加 Page Retriever。
4. Operation/Argument 错误最多：细化操作 ontology 与参数校验。
5. Evidence 已正确但答案仍错：修复 Tool 或 Normalizer。
6. 基线稳定后仍有可验证歧义：增加窄职责 Verifier。
