from table_qa_agent.agents import (
    ComputeAgent,
    ExtractAgent,
    StructureAgent,
    VisualAttributeAgent,
    build_specialist_router,
)
from table_qa_agent.orchestration import TaskPlanner
from table_qa_agent.schemas import QuestionRecord


class FakeClient:
    pass


def _question(
    text: str,
    *,
    question_type: str = "extract",
    answer_format: str = "string",
) -> QuestionRecord:
    return QuestionRecord.model_validate(
        {
            "id": 1,
            "file_name": "table.png",
            "question_type": question_type,
            "question": text,
            "answer_format": answer_format,
        }
    )


def test_router_uses_distinct_workflow_agent_classes_and_tool_whitelists() -> None:
    router = build_specialist_router(
        FakeClient(),  # type: ignore[arg-type]
        validate_evidence=False,
        force_answer=True,
        enable_tool_calls=True,
    )
    planner = TaskPlanner()

    extract = router.route(planner.plan(_question("提取营业收入")))
    compute = router.route(
        planner.plan(
            _question(
                "营业收入增长率是多少？",
                question_type="thinking",
                answer_format="number",
            )
        )
    )
    structure = router.route(
        planner.plan(
            _question(
                "恢复表格结构",
                question_type="structure",
                answer_format="json",
            )
        )
    )
    visual = router.route(planner.plan(_question("背景颜色是什么？")))

    assert isinstance(extract, ExtractAgent)
    assert isinstance(compute, ComputeAgent)
    assert isinstance(structure, StructureAgent)
    assert isinstance(visual, VisualAttributeAgent)
    assert extract.enable_tool_calls is True
    assert compute.enable_tool_calls is True
    assert structure.enable_tool_calls is True
    assert visual.enable_tool_calls is False
