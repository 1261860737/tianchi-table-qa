import pytest
from pydantic import ValidationError

from table_qa_agent.client import ModelCompletion
from table_qa_agent.ocr import VisionOCRBackend
from table_qa_agent.orchestration import TaskPlanner
from table_qa_agent.retrieval import PageRegionLocator
from table_qa_agent.schemas import QuestionRecord, RegionRef, TokenUsage


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.kwargs: dict[str, object] = {}

    def complete(self, **_: object) -> ModelCompletion:
        self.kwargs = _
        return ModelCompletion(text=self.text, usage=TokenUsage(total_tokens=5))


def _question() -> QuestionRecord:
    return QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="extract",
        question="目标字段是多少？",
        answer_format="number",
    )


def test_locator_parses_normalized_regions() -> None:
    client = FakeClient(
        '{"regions":[{"page":1,"bbox":[0.1,0.2,0.8,0.7],"confidence":0.9}]}'
    )
    locator = PageRegionLocator(client, max_regions=2)  # type: ignore[arg-type]

    regions, usage = locator.locate(
        _question(),
        TaskPlanner().plan(_question()),
        [],
        failure_reason="文字过小",
    )

    assert regions[0].bbox == (0.1, 0.2, 0.8, 0.7)
    assert usage.total_tokens == 5


def test_region_rejects_invalid_normalized_bbox() -> None:
    with pytest.raises(ValidationError, match="bbox"):
        RegionRef(page=1, bbox=(0.8, 0.2, 0.1, 0.7))


def test_vision_ocr_returns_candidates_without_answering() -> None:
    backend = VisionOCRBackend(  # type: ignore[arg-type]
        FakeClient('{"blocks":[{"text":"营业收入 1,200","confidence":0.95}]}')
    )

    result, usage = backend.recognize(_question(), [])

    assert result.as_prompt_text() == "营业收入 1,200"
    assert usage.total_tokens == 5


def test_structure_locator_and_specialist_share_scope_contract() -> None:
    from table_qa_agent.prompts import build_specialist_system_prompt
    from table_qa_agent.structure.contract import STRUCTURE_SCOPE_CONTRACT

    question = _question().model_copy(update={
        "question_type": "structure", "answer_format": "json", "question": "恢复表头",
    })
    client = FakeClient('{"regions":[{"page":1,"bbox":[0,0,1,1]}]}')
    PageRegionLocator(client).locate(
        question, TaskPlanner().plan(question), [], failure_reason="补读",
    )
    assert STRUCTURE_SCOPE_CONTRACT in client.kwargs["system_prompt"]
    assert STRUCTURE_SCOPE_CONTRACT in build_specialist_system_prompt("structure")
    assert STRUCTURE_SCOPE_CONTRACT not in build_specialist_system_prompt("extract")
