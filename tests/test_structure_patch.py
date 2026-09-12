from copy import deepcopy

import pytest

from table_qa_agent.structure.patch import apply_structure_patch, conflict_indices


@pytest.fixture
def candidate() -> dict:
    return {"row_count": 3, "col_count": 3, "cells": [
        {"text": "父级", "row": 0, "col": 0, "rowspan": 2, "colspan": 2},
        {"text": "子级", "row": 1, "col": 0, "rowspan": 1, "colspan": 1},
        {"text": "其他", "row": 0, "col": 2, "rowspan": 1, "colspan": 1},
    ]}


def test_geometry_patch_preserves_text_other_cells_and_source(candidate: dict) -> None:
    original = deepcopy(candidate)
    assert conflict_indices(candidate) == [0, 1]
    output = apply_structure_patch(candidate, {"updates": [
        {"index": 0, "row": 0, "col": 0, "rowspan": 1, "colspan": 2},
    ]}).model_dump()
    assert candidate == original
    assert output["cells"][1:] == original["cells"][1:]
    assert output["cells"][0]["text"] == "父级"


@pytest.mark.parametrize("change", [
    {"index": 2, "row": 2, "col": 2, "rowspan": 1, "colspan": 1},
    {"index": 0, "row": 0, "col": 0, "rowspan": 2, "colspan": 2},
    {"index": 0, "row": 0, "col": 0, "rowspan": 1, "colspan": 2, "text": "删除"},
])
def test_invalid_or_out_of_scope_patch_rejected(candidate: dict, change: dict) -> None:
    with pytest.raises(ValueError):
        apply_structure_patch(candidate, {"updates": [change]})


def test_empty_or_duplicate_patch_rejected(candidate: dict) -> None:
    change = {"index": 0, "row": 0, "col": 0, "rowspan": 1, "colspan": 2}
    for updates in ([], [change, change]):
        with pytest.raises(ValueError):
            apply_structure_patch(candidate, {"updates": updates})


def test_pipeline_applies_patch_and_records_proposal(tmp_path, candidate: dict) -> None:
    import json

    from table_qa_agent.agent import AgentOutput
    from table_qa_agent.client import ModelCompletion
    from table_qa_agent.config import DocumentConfig, RuntimeConfig
    from table_qa_agent.dataset import DocumentResolver
    from table_qa_agent.documents import DocumentProcessor
    from table_qa_agent.pipeline import BaselinePipeline
    from table_qa_agent.schemas import (
        EvidenceItem,
        EvidenceResponse,
        OperationSpec,
        QuestionRecord,
        RunResult,
        TaskPlan,
        TokenUsage,
    )

    class Agent:
        calls = 0

        def propose_structure_patch(self, question, value, images, error):
            self.calls += 1
            assert images
            return ModelCompletion(text=json.dumps({"updates": [
                {"index": 0, "row": 0, "col": 0, "rowspan": 1, "colspan": 2},
            ]}), usage=TokenUsage(total_tokens=7))

    agent = Agent()
    pipeline = BaselinePipeline(
        resolver=DocumentResolver(tmp_path), agent=agent,
        processor=DocumentProcessor(DocumentConfig(cache_dir=tmp_path / "cache")),
        runtime=RuntimeConfig(log_dir=tmp_path / "logs"),
    )
    question = QuestionRecord(
        id=1, file_name="table.png", question_type="structure",
        question="恢复表头", answer_format="json",
    )
    result = RunResult(question_id=1, question=question.question, document_id="table.png")
    pipeline._execute_agent_output(
        result=result, question=question,
        plan=TaskPlan(task_kind="structure_recover", specialist="structure"), agent=agent,
        structure_images=[{"type": "text", "text": "测试上下文"}],
        agent_output=AgentOutput(evidence=EvidenceResponse(
            question_type="structure", evidence=[EvidenceItem(value=candidate)],
            operation=OperationSpec(name="lookup", arguments={"value": {"evidence_index": 0}}),
        ), raw_text="{}", usage=TokenUsage()),
    )
    assert agent.calls == 1
    assert result.recovery_attempts[-1].action == "patch_structure"
    assert result.recovery_attempts[-1].succeeded
    assert result.recovery_attempts[-1].raw_output
    assert json.loads(result.final_answer)["cells"][0]["rowspan"] == 1
    assert candidate["cells"][0]["rowspan"] == 2
