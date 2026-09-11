"""Evidence Agent：提示词、模型调用、JSON 修复和协议校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from json_repair import loads as repair_json_loads

from table_qa_agent.capabilities import AGENT_TOOL_DEFINITIONS, validate_tool_call
from table_qa_agent.client import ModelCompletion, ModelToolCall, OpenAICompatibleVLClient
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
        try:
            evidence = parse_evidence_response(completion.text)
        except Exception as exc:
            raise EvidenceValidationError(
                "模型输出无法解析为 Evidence JSON",
                raw_text=completion.text,
                usage=completion.usage,
            ) from exc

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


def parse_evidence_response(text: str) -> EvidenceResponse:
    """解析并按本地协议轻量修复模型 Evidence JSON，供运行和日志重放复用。"""

    parsed = repair_json_loads(text)
    return EvidenceResponse.model_validate(parsed)
