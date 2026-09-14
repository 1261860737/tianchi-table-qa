import json

import pytest

from table_qa_agent.agent import EvidenceAgent, EvidenceValidationError
from table_qa_agent.client import ModelCompletion, ModelToolCall
from table_qa_agent.prompts import build_question_prompt, build_specialist_system_prompt
from table_qa_agent.schemas import QuestionRecord, TokenUsage


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, **_: object) -> ModelCompletion:
        return ModelCompletion(text=self.text, usage=TokenUsage(total_tokens=42))


class ToolCallingFakeClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def complete(self, **kwargs: object) -> ModelCompletion:
        self.kwargs = kwargs
        return ModelCompletion(
            text="",
            usage=TokenUsage(total_tokens=7),
            tool_calls=[
                ModelToolCall(
                    id="call_1",
                    name="inspect_table_region",
                    arguments={"query": "华东收入", "reason": "数字太小"},
                )
            ],
        )


def _question() -> QuestionRecord:
    return QuestionRecord(
        id=1,
        file_name="001.pdf",
        question_type="extract",
        question="华东收入是多少？",
        answer_format="number",
    )


def test_agent_repairs_minor_json_error() -> None:
    raw = """{
      "status": "success",
      "question_type": "extract",
      "evidence": [{"value_raw": "120", "value": 120, "page": 1}],
      "operation": {"name": "lookup", "arguments": {"value": 120}},
      "output": {"type": "number"},
    }"""
    output = EvidenceAgent(FakeClient(raw)).run(_question(), [])  # type: ignore[arg-type]
    assert output.evidence.operation is not None
    assert output.evidence.operation.arguments["value"] == 120
    assert output.usage.total_tokens == 42


def test_structure_evidence_accepts_header_arrays() -> None:
    raw = """{
      "status": "success",
      "question_type": "structure",
      "evidence": [{
        "column_header": ["公司名称", "工具名称", "国别"],
        "value_raw": "OpenAI AgentKit 美国",
        "value": ["OpenAI", "AgentKit", "美国"],
        "page": 1
      }],
      "operation": {
        "name": "lookup",
        "arguments": {"value": ["OpenAI", "AgentKit", "美国"]}
      },
      "output": {"type": "json"}
    }"""
    output = EvidenceAgent(FakeClient(raw)).run(_question(), [])  # type: ignore[arg-type]
    assert output.evidence.evidence[0].column_header == ["公司名称", "工具名称", "国别"]


def test_agent_preserves_invalid_raw_output_for_debugging() -> None:
    raw = '{"status":"success","question_type":"extract","evidence":[]}'
    with pytest.raises(EvidenceValidationError) as caught:
        EvidenceAgent(FakeClient(raw)).run(_question(), [])  # type: ignore[arg-type]
    assert caught.value.raw_text == raw


def test_agent_tolerates_unicode_evidence_id_and_slight_bbox_overflow() -> None:
    raw = """{
      "status": "success",
      "question_type": "extract",
      "evidence": [{
        "id": "ev_nivå_3",
        "value_raw": "93",
        "value": 93,
        "bbox": [0.82, 0.96, 0.98, 1.04],
        "page": 1
      }],
      "operation": {"name": "lookup", "arguments": {"value": {"evidence_index": 0}}},
      "output": {"type": "number"}
    }"""

    output = EvidenceAgent(FakeClient(raw)).run(_question(), [])  # type: ignore[arg-type]

    assert output.evidence.evidence[0].id == "ev_nivå_3"
    assert output.evidence.evidence[0].bbox == (0.82, 0.96, 0.98, 1.0)


@pytest.mark.parametrize("bbox", [
    [0, 0, 0, 0],
    [0.4, 0.2, 0.1, 0.8],
    ["bad", 0, 1, 1],
    [0, 1, 1],
])
def test_agent_treats_invalid_evidence_bbox_as_missing(bbox: list[object]) -> None:
    raw = json.dumps({
        "status": "success",
        "question_type": "extract",
        "evidence": [{"value_raw": "120", "value": 120, "bbox": bbox}],
        "operation": {"name": "lookup", "arguments": {
            "value": {"evidence_index": 0},
        }},
        "output": {"type": "number"},
    })

    output = EvidenceAgent(FakeClient(raw)).run(_question(), [])  # type: ignore[arg-type]

    assert output.evidence.evidence[0].value == 120
    assert output.evidence.evidence[0].bbox is None


def test_agent_treats_operation_with_steps_as_pipeline() -> None:
    raw = """{
      "status": "success",
      "question_type": "thinking",
      "evidence": [
        {"value_raw": "2", "value": 2, "page": 1},
        {"value_raw": "3", "value": 3, "page": 1}
      ],
      "operation": {
        "name": "add",
        "arguments": {"a": {"evidence_index": 0}, "b": {"evidence_index": 1}},
        "steps": [{
          "id": "total",
          "name": "add",
          "arguments": {"a": {"evidence_index": 0}, "b": {"evidence_index": 1}}
        }]
      },
      "output": {"type": "number"}
    }"""

    output = EvidenceAgent(FakeClient(raw)).run(_question(), [])  # type: ignore[arg-type]

    assert output.evidence.operation is not None
    assert output.evidence.operation.name == "pipeline"


def test_force_answer_prompt_disallows_refusal_and_keeps_format_contract() -> None:
    prompt = build_question_prompt(_question(), force_answer=True)
    system_prompt = build_specialist_system_prompt("extract", force_answer=True)

    assert "优先调用工具" in prompt
    assert "必须返回 status=success" in prompt
    assert "最终值必须非空，且严格满足题目的 answer_format" in prompt
    assert "启用 format-only 时不得走拒答" in system_prompt
    assert "format-only 答案策略（优先于上面的证据不足与拒答规则）" in system_prompt
    assert "Evidence 可选且不作为拒答条件" in system_prompt


def test_extract_agent_can_request_shared_table_inspection_tool() -> None:
    client = ToolCallingFakeClient()
    output = EvidenceAgent(
        client,  # type: ignore[arg-type]
        specialist="extract",
        enable_tool_calls=True,
    ).run(_question(), [{"type": "image_url"}])

    assert output.tool_calls[0].name == "inspect_table_region"
    assert output.evidence.status == "insufficient"
    assert client.kwargs["tools"]


def test_extract_agent_accepts_direct_answer_without_lookup_operation() -> None:
    raw = """{
      "status": "success",
      "question_type": "extract",
      "evidence": [{"value_raw": "120", "value": 120, "page": 1}],
      "operation": null,
      "direct_answer": 120,
      "output": {"type": "number"}
    }"""

    output = EvidenceAgent(FakeClient(raw), specialist="extract").run(  # type: ignore[arg-type]
        _question(), []
    )

    assert output.evidence.operation is None
    assert output.evidence.direct_answer == 120


def test_experimental_agent_allows_direct_literal_without_evidence() -> None:
    raw = """{
      "status": "success",
      "question_type": "extract",
      "evidence": [],
      "operation": {"name": "lookup", "arguments": {"value": 42}},
      "output": {"type": "number"}
    }"""

    output = EvidenceAgent(
        FakeClient(raw),  # type: ignore[arg-type]
        validate_evidence=False,
        force_answer=True,
    ).run(_question(), [])

    assert output.evidence.evidence == []
    assert output.evidence.operation is not None
    assert output.evidence.operation.arguments["value"] == 42
