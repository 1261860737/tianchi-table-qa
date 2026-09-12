"""Evidence Agent：提示词、模型调用、JSON 修复和协议校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from json_repair import loads as repair_json_loads

from table_qa_agent.capabilities import AGENT_TOOL_DEFINITIONS, validate_tool_call
from table_qa_agent.client import ModelCompletion, ModelToolCall, OpenAICompatibleVLClient
from table_qa_agent.normalizer import normalize_answer, validate_answer_text
from table_qa_agent.prompts import (
    FORCE_ANSWER_SYSTEM_APPENDIX,
    OPERATION_REPAIR_PROMPT,
    SYSTEM_PROMPT,
    build_question_prompt,
    build_specialist_system_prompt,
)
from table_qa_agent.schemas import (
    AnswerProjection,
    EvidenceResponse,
    OperationSpec,
    QuestionRecord,
    SpecialistName,
    TaskPlan,
    TokenUsage,
)
from table_qa_agent.structure.contract import STRUCTURE_SCOPE_CONTRACT
from table_qa_agent.structure.patch import conflict_indices


class EvidenceValidationError(ValueError):
    """模型输出虽可解析，但不满足 Evidence 协议。"""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.usage = usage


@dataclass(slots=True)
class AgentOutput:
    evidence: EvidenceResponse
    raw_text: str
    usage: TokenUsage
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class EvidenceAgent:
    """一次模型调用完成 Query-aware Cell Grounding。"""

    def __init__(
        self,
        client: OpenAICompatibleVLClient,
        *,
        specialist: SpecialistName | None = None,
        validate_evidence: bool = True,
        force_answer: bool = False,
        enable_tool_calls: bool = False,
    ) -> None:
        self.client = client
        self.specialist = specialist
        self.validate_evidence = validate_evidence
        self.force_answer = force_answer
        self.enable_tool_calls = enable_tool_calls

    def run(
        self,
        question: QuestionRecord,
        image_content: list[dict[str, object]],
        *,
        plan: TaskPlan | None = None,
        ocr_text: str | None = None,
        recovery_context: str | None = None,
    ) -> AgentOutput:
        specialist = self.specialist or (plan.specialist if plan is not None else None)
        if specialist is not None:
            system_prompt = build_specialist_system_prompt(
                specialist,
                force_answer=self.force_answer,
            )
        else:
            system_prompt = SYSTEM_PROMPT
            if self.force_answer:
                system_prompt += "\n\n" + FORCE_ANSWER_SYSTEM_APPENDIX
        tools = None
        if (
            self.enable_tool_calls
            and specialist in {"extract", "compute", "structure"}
            and recovery_context is None
        ):
            tools = AGENT_TOOL_DEFINITIONS
        completion: ModelCompletion = self.client.complete(
            system_prompt=system_prompt,
            user_text=build_question_prompt(
                question,
                plan,
                ocr_text=ocr_text,
                recovery_context=recovery_context,
                force_answer=self.force_answer,
            ),
            image_content=image_content,
            tools=tools,
        )
        if completion.tool_calls:
            if len(completion.tool_calls) != 1:
                raise EvidenceValidationError(
                    "每轮只允许请求一个 Agent 工具",
                    raw_text=completion.text,
                    usage=completion.usage,
                )
            try:
                validate_tool_call(completion.tool_calls[0])
            except Exception as exc:
                raise EvidenceValidationError(
                    f"Agent 工具调用参数无效: {exc}",
                    raw_text=completion.text,
                    usage=completion.usage,
                ) from exc
            requested = completion.tool_calls[0]
            raw_text = json.dumps(
                {
                    "tool_call": {
                        "id": requested.id,
                        "name": requested.name,
                        "arguments": requested.arguments,
                    }
                },
                ensure_ascii=False,
            )
            return AgentOutput(
                evidence=EvidenceResponse(
                    status="insufficient",
                    question_type=question.question_type,
                    reason=f"请求共享工具 {requested.name}",
                ),
                raw_text=raw_text,
                usage=completion.usage,
                tool_calls=completion.tool_calls,
            )
        # 抽取答案先按最终合同检查；辅助 Evidence 不具有否决权。
        if specialist in {"extract", "visual_attribute"} and question.answer_format != "json":
            direct = parse_direct_response(completion.text, question)
            if direct is not None:
                evidence, warnings = direct
                return AgentOutput(
                    evidence=evidence, raw_text=completion.text,
                    usage=completion.usage, warnings=warnings,
                )
        protocol_warnings: list[str] = []
        try:
            evidence = parse_evidence_response(completion.text)
            if specialist == "compute" and evidence.operation is None:
                raise ValueError("计算响应缺少 operation，不能使用模型最终答案替代计算")
        except Exception as exc:
            if specialist != "compute" or recovery_context is None:
                raise EvidenceValidationError(
                    f"模型输出无法解析为 Evidence JSON: {exc}",
                    raw_text=completion.text, usage=completion.usage,
                ) from exc
            # 只在已经进入补读恢复的计算题增加一次协议修复，不递归重试。
            initial = completion
            retry = self.client.complete(
                system_prompt=system_prompt,
                user_text=build_question_prompt(question, plan, ocr_text=ocr_text)
                + "\n上次响应缺少必要计算协议。请重新从图片读取操作数，不把上次答案当作证据。"
                + "必须返回 question_type、evidence、operation 和 output；禁止仅返回 answer。"
                + "选择最简单的 Python 操作并引用证据，禁止自行计算最终值。",
                image_content=image_content,
            )
            usage = TokenUsage(**{
                key: (getattr(initial.usage, key) or 0) + (getattr(retry.usage, key) or 0)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            })
            completion = ModelCompletion(text=retry.text, usage=usage)
            try:
                evidence = parse_evidence_response(completion.text)
                if evidence.operation is None or not evidence.evidence:
                    raise ValueError("协议修复后仍缺少 Evidence 或 Operation")
            except Exception as retry_error:
                raise EvidenceValidationError(
                    f"计算协议修复失败: {retry_error}",
                    raw_text=completion.text, usage=usage,
                ) from retry_error
            protocol_warnings.append(
                f"计算协议缺失，已重新读取操作数并修复一次；原始响应：{initial.text}"
            )

        if self.validate_evidence and evidence.status == "success":
            if evidence.operation is None and evidence.direct_answer is None:
                raise EvidenceValidationError(
                    "status=success 时 operation 和 direct_answer 不能同时为空",
                    raw_text=completion.text,
                    usage=completion.usage,
                )
            if not evidence.evidence:
                raise EvidenceValidationError(
                    "status=success 时 evidence 不能为空",
                    raw_text=completion.text,
                    usage=completion.usage,
                )
        elif self.validate_evidence and (
            evidence.operation is not None or evidence.direct_answer is not None
        ):
            raise EvidenceValidationError(
                "status 为 ambiguous/insufficient 时 operation/direct_answer 必须为空",
                raw_text=completion.text,
                usage=completion.usage,
            )
        if self.validate_evidence and (
            specialist is not None
            and evidence.task_plan is not None
            and evidence.task_plan.specialist != specialist
        ):
            raise EvidenceValidationError(
                "专家输出的 task_plan.specialist 与路由结果不一致",
                raw_text=completion.text,
                usage=completion.usage,
            )

        return AgentOutput(
            evidence=evidence,
            raw_text=completion.text,
            usage=completion.usage,
            warnings=protocol_warnings,
        )

    def propose_structure_patch(
        self, question: QuestionRecord, value: dict[str, object],
        image_content: list[dict[str, object]], error: Exception,
    ) -> ModelCompletion:
        return self.client.complete(
            system_prompt=STRUCTURE_SCOPE_CONTRACT + "\n"
            "你是结构几何修复器。对照图片，只修正 allowed_indices 中单元格的位置或跨度。"
            "父级表头有子级并不表示父级占据子级所在行；以实际可见边界为准。"
            "禁止修改文本、其他单元格和整表尺寸，禁止删格或新增格。"
            '只返回 JSON：{"updates":[{"index":0,"row":0,"col":0,"rowspan":1,"colspan":1}]}。'
            '无法在此边界内确定修复时返回 {"updates":[]}，不要强行猜测。',
            user_text=json.dumps({
                "question": question.question, "candidate": value,
                "allowed_indices": conflict_indices(value), "error": str(error),
            }, ensure_ascii=False),
            image_content=image_content,
        )

    def repair_operation(
        self,
        question: QuestionRecord,
        plan: TaskPlan,
        evidence: EvidenceResponse,
        error: Exception,
    ) -> AgentOutput:
        payload = {
            "question": question.model_dump(mode="json"),
            "task_plan": plan.model_dump(mode="json"),
            "evidence": [item.model_dump(mode="json") for item in evidence.evidence],
            "current_operation": (
                evidence.operation.model_dump(mode="json")
                if evidence.operation is not None
                else None
            ),
            "error": str(error),
        }
        completion = self.client.complete(
            system_prompt=OPERATION_REPAIR_PROMPT,
            user_text=json.dumps(payload, ensure_ascii=False),
            image_content=[],
        )
        try:
            parsed = repair_json_loads(completion.text)
            operation = OperationSpec.model_validate(parsed["operation"])
            projection_raw = parsed.get("answer_projection")
            projection = (
                AnswerProjection.model_validate(projection_raw)
                if projection_raw is not None
                else evidence.answer_projection
            )
            repaired = evidence.model_copy(
                update={
                    "operation": operation,
                    "answer_projection": projection,
                    "output": parsed.get("output") or evidence.output,
                }
            )
        except Exception as exc:
            raise EvidenceValidationError(
                "Operation 修复输出无法解析",
                raw_text=completion.text,
                usage=completion.usage,
            ) from exc
        return AgentOutput(evidence=repaired, raw_text=completion.text, usage=completion.usage)


def parse_direct_response(
    text: str, question: QuestionRecord,
) -> tuple[EvidenceResponse, list[str]] | None:
    """只保护符合最终格式的直接答案，原始辅助信息仍保存在 raw_text 中。"""
    try:
        parsed = repair_json_loads(text)
        if not isinstance(parsed, dict) or parsed.get("direct_answer") is None:
            return None
        # direct_answer 是最终值，不使用模型额外的投影、精度等指令改写。
        answer = normalize_answer(parsed["direct_answer"], question.answer_format, {})
        if validate_answer_text(answer, question, allow_blank=False):
            return None
    except Exception:
        return None
    warnings: list[str] = []
    try:
        response = EvidenceResponse.model_validate(parsed)
    except Exception:
        response = EvidenceResponse(question_type=question.question_type)
        warnings.append("辅助 Evidence/计划协议无效，仅保留原始日志；直接答案未被否决")
    if not response.evidence:
        warnings.append("直接答案缺少可解析 Evidence，仅记录，不触发重答")
    return response.model_copy(update={
        "status": "success", "direct_answer": parsed["direct_answer"], "output": {},
    }), warnings


def parse_evidence_response(text: str) -> EvidenceResponse:
    """解析并按本地协议轻量修复模型 Evidence JSON，供运行和日志重放复用。"""

    parsed = repair_json_loads(text)
    return EvidenceResponse.model_validate(parsed)
