import pytest

from table_qa_agent.client import ModelCompletion, ModelToolCall
from table_qa_agent.orchestration import FunctionCallingIntentPlanner, TaskPlanner
from table_qa_agent.schemas import QuestionRecord, TokenUsage


class IntentClient:
    def __init__(self, arguments: dict[str, object]) -> None:
        self.arguments = arguments
        self.kwargs: dict[str, object] = {}

    def complete(self, **kwargs: object) -> ModelCompletion:
        self.kwargs = kwargs
        return ModelCompletion(
            text="",
            usage=TokenUsage(total_tokens=9),
            tool_calls=[
                ModelToolCall(
                    id="intent_1",
                    name="create_task_plan",
                    arguments=self.arguments,
                )
            ],
        )


def _question(text: str, *, question_type: str = "thinking", answer_format: str = "number"):
    return QuestionRecord.model_validate(
        {
            "id": 1,
            "file_name": "table.png",
            "question_type": question_type,
            "question": text,
            "answer_format": answer_format,
        }
    )


def test_duration_route_declares_typed_time_fields() -> None:
    plan = TaskPlanner().plan(_question("从 12:00 到 14:00 持续多少分钟？"))

    assert plan.specialist == "compute"
    assert plan.task_kind == "compute_duration"
    assert plan.preferred_operation == "duration"
    assert [field.role for field in plan.required_fields] == ["start_time", "end_time"]


def test_argmax_label_route_uses_answer_projection() -> None:
    plan = TaskPlanner().plan(
        _question("本期支付租金最高的出租方是哪一个？", answer_format="string")
    )

    assert plan.task_kind == "compute_arg_extreme"
    assert plan.preferred_operation == "argmax"
    assert plan.answer_projection.mode == "path"
    assert plan.answer_projection.path == ["label"]


@pytest.mark.parametrize(
    "text",
    [
        "固定资产期末账面价值最高的资产类别是什么？",
        "销售额最低的是哪个产品？",
        "找出价格最大的公司名称。",
        "哪种材料的库存量最小？",
    ],
)
def test_extreme_string_questions_project_only_label(text: str) -> None:
    plan = TaskPlanner().plan(_question(text, answer_format="string"))

    assert plan.task_kind == "compute_arg_extreme"
    assert plan.answer_projection.mode == "path"
    assert plan.answer_projection.path == ["label"]


def test_extreme_number_question_returns_the_numeric_extreme() -> None:
    plan = TaskPlanner().plan(_question("所有产品中最高价格是多少？"))

    assert plan.task_kind == "compute_extreme"
    assert plan.preferred_operation == "max"
    assert plan.answer_projection.mode == "identity"


def test_later_row_route_projects_requested_identifier() -> None:
    plan = TaskPlanner().plan(_question("请找出 Year 较晚的一行的 Season Number。"))

    assert plan.task_kind == "compute_arg_extreme"
    assert plan.preferred_operation == "argmax"
    assert plan.answer_projection.path == ["label"]


def test_multi_field_array_route_wins_over_extreme_keyword() -> None:
    plan = TaskPlanner().plan(
        _question(
            "请按表中顺序列出：最低分、最低分姓名、记录人数。",
            answer_format="json_array",
        )
    )

    assert plan.mode == "multi_field"
    assert plan.answer_projection.mode == "identity"


def test_structure_and_visual_routes_are_capability_based() -> None:
    structure = TaskPlanner().plan(
        _question("恢复表格结构", question_type="structure", answer_format="json")
    )
    visual = TaskPlanner().plan(
        _question("背景颜色是什么？", question_type="extract", answer_format="string")
    )

    assert (structure.specialist, structure.mode) == ("structure", "recover")
    assert visual.specialist == "visual_attribute"


def test_function_calling_intent_planner_returns_policy_checked_plan() -> None:
    client = IntentClient(
        {
            "task_kind": "compute_arithmetic",
            "specialist": "compute",
            "mode": "ratio",
            "required_fields": [
                {"name": "Net Income", "role": "measure", "value_type": "number"},
                {"name": "Total Revenue", "role": "measure", "value_type": "number"},
            ],
            "preferred_operation": "divide",
            "answer_projection": {"mode": "identity", "path": [], "fields": []},
            "needs_localization": True,
            "needs_ocr": False,
        }
    )
    planner = FunctionCallingIntentPlanner(client)  # type: ignore[arg-type]

    result = planner.plan(_question("请用 Net Income 除以 Total Revenue。"))

    assert result.plan.specialist == "compute"
    assert result.plan.preferred_operation is None
    assert result.plan.answer_projection.mode == "identity"
    assert result.usage.total_tokens == 9
    assert client.kwargs["tool_choice"] == "required"
    assert client.kwargs["image_content"] == []


def test_function_calling_intent_planner_falls_back_on_policy_violation() -> None:
    client = IntentClient(
        {
            "task_kind": "extract_scalar",
            "specialist": "extract",
            "mode": "scalar",
            "required_fields": [],
            "preferred_operation": "lookup",
            "answer_projection": {"mode": "identity", "path": [], "fields": []},
            "needs_localization": True,
            "needs_ocr": False,
        }
    )
    planner = FunctionCallingIntentPlanner(client)  # type: ignore[arg-type]

    result = planner.plan(
        _question("恢复表格结构", question_type="structure", answer_format="json")
    )

    assert result.plan.specialist == "structure"
    assert result.plan.task_kind == "structure_recover"
    assert any("回退本地规则" in warning for warning in result.warnings)


def test_intent_tool_no_longer_requests_execution_details() -> None:
    from table_qa_agent.orchestration.planner import INTENT_PLANNER_TOOLS

    client = IntentClient({
        "task_kind": "compute_arithmetic", "specialist": "compute", "mode": "ratio",
        "required_fields": [], "needs_localization": False, "needs_ocr": False,
    })
    result = FunctionCallingIntentPlanner(client).plan(_question("收入占比是多少？"))
    assert result.warnings == []
    assert result.plan.preferred_operation is None
    schema = INTENT_PLANNER_TOOLS[0]["function"]["parameters"]
    for field in ("preferred_operation", "answer_projection"):
        assert field not in schema["properties"]
        assert field not in schema["required"]
        assert field not in client.kwargs["user_text"]


def test_legacy_model_projection_cannot_override_expert() -> None:
    client = IntentClient({
        "task_kind": "compute_boolean", "specialist": "compute", "mode": "multi_field",
        "required_fields": [], "preferred_operation": "pipeline",
        "answer_projection": {"mode": "fields", "fields": ["item"]},
    })
    result = FunctionCallingIntentPlanner(client).plan(
        _question("哪些项目有变动？", answer_format="json_array")
    )
    assert result.warnings == []
    assert result.plan.preferred_operation is None
    assert result.plan.answer_projection.mode == "identity"
