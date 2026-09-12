"""有效答案不能被无关的辅助字段否决。"""

import json

import pytest

from table_qa_agent.agent import EvidenceAgent
from table_qa_agent.client import ModelCompletion
from table_qa_agent.schemas import QuestionRecord, TokenUsage


class Client:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, **kwargs: object) -> ModelCompletion:
        return ModelCompletion(text=json.dumps(self.payload), usage=TokenUsage())


@pytest.mark.parametrize("auxiliary", [
    {"evidence": []},
    {"evidence": [{"value": "英文", "bbox": [0, 1.12, 0.5, 1.22]}]},
    {"evidence": [{"id": "same", "value": 1}, {"id": "same", "value": 2}]},
    {"operation": {"name": "unsupported"}, "task_plan": {"mode": "single_value"}},
])
@pytest.mark.parametrize("strict", [True, False])
def test_valid_direct_answer_survives_auxiliary_failure(auxiliary: dict, strict: bool) -> None:
    question = QuestionRecord(
        id=838, file_name="001.png", question_type="extract",
        question="提取 email_id", answer_format="string",
    )
    client = Client({
        "question_type": "extract", "direct_answer": "example@gmail.com",
        "status": "insufficient", **auxiliary,
    })
    output = EvidenceAgent(
        client, specialist="extract", validate_evidence=strict,
    ).run(question, [])
    assert output.evidence.status == "success"
    assert output.evidence.direct_answer == "example@gmail.com"
    assert output.warnings


def test_compute_does_not_use_answer_first_escape_hatch() -> None:
    from table_qa_agent.agent import EvidenceValidationError

    question = QuestionRecord(
        id=1, file_name="001.png", question_type="thinking",
        question="两项之和", answer_format="number",
    )
    with pytest.raises(EvidenceValidationError):
        EvidenceAgent(Client({
            "question_type": "thinking", "direct_answer": 42, "evidence": [],
        }), specialist="compute").run(question, [])


@pytest.mark.parametrize("specialist", ["extract", "visual_attribute"])
def test_direct_array_survives_invalid_auxiliary_operation(specialist: str) -> None:
    question = QuestionRecord(id=1, file_name="a.png", question_type="extract",
                              question="依次读取两个字段", answer_format="json_array")
    output = EvidenceAgent(Client({
        "status": "success", "direct_answer": ["甲", "乙"],
        "operation": {"name": "count", "arguments": {"source": 7}},
        "evidence": [{"bbox": [0, 2, 1, 3]}],
    }), specialist=specialist).run(question, [])
    assert output.evidence.direct_answer == ["甲", "乙"]


def test_compute_protocol_repair_requires_operands_and_combines_usage() -> None:
    class SequenceClient:
        calls = 0

        def complete(self, **kwargs: object) -> ModelCompletion:
            self.calls += 1
            text = '{"answer":999}' if self.calls == 1 else json.dumps({
                "question_type": "thinking", "evidence": [{"value": 5}, {"value": 2}],
                "operation": {"name": "subtract", "arguments": {
                    "a": {"evidence_index": 0}, "b": {"evidence_index": 1},
                }},
            })
            return ModelCompletion(text=text, usage=TokenUsage(total_tokens=5))

    client = SequenceClient()
    question = QuestionRecord(
        id=1, file_name="001.png", question_type="thinking",
        question="差值", answer_format="number",
    )
    output = EvidenceAgent(client, specialist="compute").run(
        question, [], recovery_context="协议缺失",
    )
    assert client.calls == 2
    assert output.usage.total_tokens == 10
    assert output.evidence.operation.name == "subtract"
    assert output.evidence.direct_answer is None
    assert output.warnings
