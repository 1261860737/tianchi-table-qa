"""基于任务本体生成通用计划，不包含数据集题目 ID 特例。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from table_qa_agent.client import OpenAICompatibleVLClient
from table_qa_agent.schemas import (
    AnswerProjection,
    QuestionRecord,
    RequiredField,
    TaskPlan,
    TokenUsage,
)

_QUOTED = re.compile(r"[“\"']([^”\"']+)[”\"']")
_TIME_QUESTION = re.compile(r"持续|时长|时间差|多少(?:分钟|小时)")
_ARG_EXTREME = re.compile(r"最高|最低|最大|最小|较晚|较早")
_LABEL_REQUEST = re.compile(
    r"哪(?:一个|个|种|类|家|位)|谁|"
    r"(?:姓名|名称|出租方|公司|产品|类别|类型|项目|科目|指标)(?:是|为)?什么|"
    r"(?:姓名|名称|出租方|公司|产品|类别|类型|项目|科目|指标)[。？?]?$"
)
_VISUAL_ATTRIBUTE = re.compile(
    r"颜色|主色|背景|(?:红|橙|黄|绿|蓝|紫|黑|白|灰)色|横向|纵向|拍摄|截图|"
    r"是否含图片|示意图|红字|表头层级|多级表头|跨页表"
)
_CATEGORY_REQUEST = re.compile(r"哪些(?:天|日期|月份|项目|产品|公司)|哪几(?:天|项|个)")

_ALLOWED_OPERATIONS = {
    "lookup",
    "list",
    "count",
    "add",
    "subtract",
    "multiply",
    "divide",
    "ratio",
    "percentage_change",
    "percentage_of_total",
    "percentage_point_difference",
    "sum",
    "average",
    "max",
    "min",
    "duration",
    "sum_durations",
    "argmax",
    "argmin",
    "boolean",
    "concat",
    "pipeline",
}

INTENT_PLANNER_TOOL_NAME = "create_task_plan"
INTENT_PLANNER_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": INTENT_PLANNER_TOOL_NAME,
            "description": (
                "分析表格问答意图，声明任务类型、专家、所需字段、确定性操作和视觉能力。"
                "这里只生成计划，不读取图片、不回答问题、不执行计算。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_kind": {
                        "type": "string",
                        "enum": [
                            "extract_scalar",
                            "extract_multi",
                            "extract_region",
                            "compute_arithmetic",
                            "compute_duration",
                            "compute_extreme",
                            "compute_arg_extreme",
                            "compute_boolean",
                            "structure_recover",
                            "structure_measure",
                            "visual_attribute",
                        ],
                    },
                    "specialist": {
                        "type": "string",
                        "enum": ["extract", "compute", "structure", "visual_attribute"],
                    },
                    "mode": {"type": "string"},
                    "required_fields": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "role": {
                                    "type": "string",
                                    "enum": [
                                        "label",
                                        "measure",
                                        "start_time",
                                        "end_time",
                                        "category",
                                        "attribute",
                                        "structure",
                                    ],
                                },
                                "value_type": {
                                    "type": "string",
                                    "enum": [
                                        "string",
                                        "number",
                                        "time",
                                        "date",
                                        "boolean",
                                        "array",
                                        "object",
                                    ],
                                },
                                "aliases": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "required": {"type": "boolean"},
                            },
                            "required": ["name", "role", "value_type"],
                            "additionalProperties": False,
                        },
                    },
                    "preferred_operation": {
                        "type": ["string", "null"],
                        "enum": [*sorted(_ALLOWED_OPERATIONS), None],
                    },
                    "answer_projection": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["identity", "path", "fields"],
                            },
                            "path": {
                                "type": "array",
                                "items": {"type": ["string", "integer"]},
                            },
                            "fields": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["mode", "path", "fields"],
                        "additionalProperties": False,
                    },
                    "needs_localization": {"type": "boolean"},
                    "needs_ocr": {"type": "boolean"},
                },
                "required": [
                    "task_kind",
                    "specialist",
                    "mode",
                    "required_fields",
                    "preferred_operation",
                    "answer_projection",
                    "needs_localization",
                    "needs_ocr",
                ],
                "additionalProperties": False,
            },
        },
    }
]

INTENT_PLANNER_PROMPT = """你是表格问答 Intent Planner，只分析题目语义和元数据。
必须调用 create_task_plan；不要回答题目，不要读取或猜测表格值，不要执行计算。

规划原则：
1. 单值原文读取用 extract_scalar；多个字段按顺序读取用 extract_multi。
2. 求和、差值、比例、平均、计数、时间差、极值和逻辑判断必须用 compute 工作流。
3. 极值询问标签用 compute_arg_extreme，询问数值用 compute_extreme。
4. 表格行列数和合并结构使用 structure；颜色、方向、布局、图片存在性使用 visual_attribute。
5. required_fields 只声明完成任务所需的原始字段，不能填写答案或中间计算结果。
6. preferred_operation 只允许选择白名单 Python 操作；复杂计算可选 pipeline。
7. needs_ocr 表示这类任务默认必须先 OCR；不确定是否需要时填 false，由专家后续请求能力。
8. answer_projection 只描述 Python 工具结果如何映射到最终答案。
"""


@dataclass(slots=True)
class PlanningResult:
    """一次意图规划的计划、用量和可审计信息。"""

    plan: TaskPlan
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw_output: str | None = None
    warnings: list[str] = field(default_factory=list)


def _named_targets(question: str) -> list[str]:
    quoted = [item.strip() for item in _QUOTED.findall(question) if item.strip()]
    if quoted:
        return quoted
    if "：" in question or ":" in question:
        tail = re.split(r"[：:]", question, maxsplit=1)[1]
        tail = re.sub(r"[。？?]+$", "", tail)
        return [item.strip() for item in re.split(r"[、，,；;]", tail) if item.strip()]
    return []


def _field(name: str, *, role: str = "measure", value_type: str = "string") -> RequiredField:
    return RequiredField.model_validate(
        {"name": name, "role": role, "value_type": value_type}
    )


class TaskPlanner:
    """用稳定元数据完成粗路由，专家 Prompt 再补充具体字段。"""

    def plan(self, question: QuestionRecord) -> TaskPlan:
        text = question.question
        targets = _named_targets(text)

        if question.question_type == "structure" and question.answer_format == "json":
            return TaskPlan(
                task_kind="structure_recover",
                specialist="structure",
                mode="recover",
                required_fields=[_field("table_structure", role="structure", value_type="object")],
                preferred_operation="lookup",
                needs_localization=True,
                needs_ocr=True,
            )

        if question.question_type == "structure":
            return TaskPlan(
                task_kind="structure_measure",
                specialist="structure",
                mode="measure",
                required_fields=[
                    _field(
                        targets[0] if targets else "structure_metric",
                        role="measure",
                        value_type="number",
                    )
                ],
                preferred_operation="count",
                needs_localization=True,
            )

        if _VISUAL_ATTRIBUTE.search(text):
            fields = targets or ["visual_attribute"]
            required_fields = [
                _field(name, role="attribute", value_type="string") for name in fields
            ]
            if _CATEGORY_REQUEST.search(text):
                required_fields.insert(
                    0,
                    _field("matching_items", role="category", value_type="array"),
                )
            return TaskPlan(
                task_kind="visual_attribute",
                specialist="visual_attribute",
                mode="multi_field" if question.answer_format == "json_array" else "scalar",
                required_fields=required_fields,
                preferred_operation="list" if question.answer_format == "json_array" else "boolean",
                needs_localization=True,
            )

        if _TIME_QUESTION.search(text):
            return TaskPlan(
                task_kind="compute_duration",
                specialist="compute",
                mode="duration",
                required_fields=[
                    _field("start_time", role="start_time", value_type="time"),
                    _field("end_time", role="end_time", value_type="time"),
                ],
                preferred_operation="duration",
                needs_localization=True,
                needs_ocr=True,
            )

        if question.answer_format == "json_array":
            fields = targets or ["item"]
            specialist = "compute" if question.question_type == "thinking" else "extract"
            return TaskPlan(
                task_kind="extract_multi" if specialist == "extract" else "compute_arithmetic",
                specialist=specialist,
                mode="multi_field",
                required_fields=[_field(name) for name in fields],
                preferred_operation="list",
                needs_localization=True,
            )

        if _ARG_EXTREME.search(text):
            # 极值工具返回 {label, value}。字符串答案通常询问“谁/哪个类别”，
            # 数值答案才询问极值本身；用 TaskPlan 做确定性投影，避免依赖模型补字段。
            wants_label = bool(
                question.answer_format == "string"
                or _LABEL_REQUEST.search(text)
                or re.search(r"对应|一行的", text)
            )
            operation = "argmin" if re.search(r"最低|最小|较早", text) else "argmax"
            return TaskPlan(
                task_kind="compute_arg_extreme" if wants_label else "compute_extreme",
                specialist="compute",
                mode="arg_extreme" if wants_label else "extreme",
                required_fields=[
                    _field("label", role="label", value_type="string"),
                    _field("value", role="measure", value_type="number"),
                ],
                preferred_operation=operation if wants_label else operation.removeprefix("arg"),
                answer_projection=(
                    AnswerProjection(mode="path", path=["label"])
                    if wants_label
                    else AnswerProjection()
                ),
                needs_localization=True,
            )

        if question.question_type == "thinking":
            return TaskPlan(
                task_kind="compute_boolean" if text.startswith("是否") else "compute_arithmetic",
                specialist="compute",
                mode="boolean" if text.startswith("是否") else "arithmetic",
                required_fields=[_field(name) for name in targets],
                needs_localization=True,
            )

        return TaskPlan(
            task_kind="extract_scalar",
            specialist="extract",
            mode="scalar",
            required_fields=[_field(name) for name in targets],
            preferred_operation="lookup",
            needs_localization=True,
        )


class FunctionCallingIntentPlanner:
    """模型声明语义计划，Python 校验策略并在失败时使用规则计划。"""

    def __init__(
        self,
        client: OpenAICompatibleVLClient,
        *,
        fallback: TaskPlanner | None = None,
        fallback_to_rules: bool = True,
    ) -> None:
        self.client = client
        self.fallback = fallback or TaskPlanner()
        self.fallback_to_rules = fallback_to_rules

    def plan(self, question: QuestionRecord) -> PlanningResult:
        fallback_plan = self.fallback.plan(question)
        usage = TokenUsage()
        raw_output: str | None = None
        payload = {
            "question_type": question.question_type,
            "question": question.question,
            "table_hint": question.table_hint,
            "answer_format": question.answer_format,
            "rule_plan_hint": fallback_plan.model_dump(mode="json"),
        }
        try:
            completion = self.client.complete(
                system_prompt=INTENT_PLANNER_PROMPT,
                user_text=json.dumps(payload, ensure_ascii=False),
                image_content=[],
                tools=INTENT_PLANNER_TOOLS,
                tool_choice="required",
            )
            usage = completion.usage
            if len(completion.tool_calls) != 1:
                raise ValueError("Intent Planner 必须且只能调用一次 create_task_plan")
            call = completion.tool_calls[0]
            if call.name != INTENT_PLANNER_TOOL_NAME:
                raise ValueError(f"Intent Planner 调用了未授权工具: {call.name}")
            raw_output = json.dumps(call.arguments, ensure_ascii=False)
            candidate = TaskPlan.model_validate(call.arguments)
            plan = _apply_plan_policy(question, candidate, fallback_plan)
            return PlanningResult(
                plan=plan,
                usage=usage,
                raw_output=raw_output,
            )
        except Exception as exc:
            if not self.fallback_to_rules:
                raise
            return PlanningResult(
                plan=fallback_plan,
                usage=usage,
                raw_output=raw_output,
                warnings=[f"模型意图规划失败，已回退本地规则 Planner: {exc}"],
            )


def _apply_plan_policy(
    question: QuestionRecord,
    candidate: TaskPlan,
    fallback: TaskPlan,
) -> TaskPlan:
    """应用无法由模型越权修改的元数据和操作白名单。"""

    if candidate.preferred_operation not in {None, *_ALLOWED_OPERATIONS}:
        raise ValueError(f"Intent Planner 选择了未授权操作: {candidate.preferred_operation}")

    if question.question_type == "structure":
        if candidate.specialist != "structure":
            raise ValueError("structure 题必须路由到 Structure Specialist")
        expected_kind = (
            "structure_recover" if question.answer_format == "json" else "structure_measure"
        )
        if candidate.task_kind != expected_kind:
            raise ValueError(f"structure 题必须使用 {expected_kind}")
    elif candidate.specialist == "structure":
        raise ValueError("非 structure 题不能进入 Structure Specialist")

    if question.answer_format == "json" and candidate.task_kind != "structure_recover":
        raise ValueError("json 答案只允许 structure_recover 工作流")
    if candidate.task_kind == "structure_recover" and question.answer_format != "json":
        raise ValueError("structure_recover 必须对应 json 答案")

    required_roles = {field.role for field in candidate.required_fields if field.required}
    if candidate.task_kind == "compute_duration" and not {
        "start_time",
        "end_time",
    }.issubset(required_roles):
        raise ValueError("duration 计划必须声明 start_time 和 end_time")
    if candidate.task_kind == "compute_arg_extreme" and not {
        "label",
        "measure",
    }.issubset(required_roles):
        raise ValueError("arg extreme 计划必须声明 label 和 measure")

    # 本地规则对硬元数据更可靠；模型计划只负责细化语义字段和任务子型。
    if fallback.specialist == "visual_attribute" and candidate.specialist not in {
        "visual_attribute",
        "compute",
    }:
        raise ValueError("视觉属性题不能降级为普通文本抽取")
    if candidate.preferred_operation == "list":
        candidate = candidate.model_copy(update={"answer_projection": AnswerProjection()})
    return candidate
