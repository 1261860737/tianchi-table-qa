"""答案接口的语义回归：使用通用合成数据，不绑定评测题号。"""

from copy import deepcopy
from decimal import Decimal

import pytest

from table_qa_agent.executor import (
    OperationExecutionError,
    execute_operation,
    operation_selects_answer,
    validate_operation_grounding,
)
from table_qa_agent.prompts import (
    FORCE_ANSWER_SYSTEM_APPENDIX,
    OPERATION_REPAIR_PROMPT,
    build_question_prompt,
    build_specialist_system_prompt,
)
from table_qa_agent.schemas import EvidenceItem, OperationSpec, QuestionRecord


def test_list_selects_entities_before_value_resolution() -> None:
    evidence = [EvidenceItem(id="a", value="红色标签", entity="周二"),
                EvidenceItem(id="b", value="绿色标签", entity="周五")]
    operation = OperationSpec(name="list", arguments={
        "values": [{"evidence_id": "b"}, {"evidence_index": 0}],
        "select_field": "entity",
    })
    before = deepcopy((operation, evidence))
    validate_operation_grounding(operation, evidence)
    assert execute_operation(operation, evidence) == ["周五", "周二"]
    assert operation_selects_answer(operation)
    assert (operation, evidence) == before


def test_list_does_not_guess_or_override_explicit_field() -> None:
    evidence = [EvidenceItem(value="标签", entity="周二")]
    assert execute_operation(OperationSpec(name="list", arguments={
        "values": [{"evidence_index": 0}],
    }), evidence) == ["标签"]
    with pytest.raises(OperationExecutionError, match="冲突"):
        execute_operation(OperationSpec(name="list", arguments={
            "values": [{"evidence_index": 0, "field": "value"}],
            "select_field": "entity",
        }), evidence)


@pytest.mark.parametrize("name,expected", [("argmax", "乙校 / 二班"),
                                          ("argmin", "甲校 / 一班")])
def test_extreme_returns_complete_entity_without_record_wrapper(name: str, expected: str) -> None:
    arguments = {"records": [{"label": "甲校 / 一班", "value": 12},
                             {"label": "乙校 / 二班", "value": 18}],
                 "return_field": "label"}
    assert execute_operation(OperationSpec(name=name, arguments=arguments)) == expected
    # 不指定新参数时仍支持历史日志的“完整记录＋投影”。
    arguments.pop("return_field")
    assert isinstance(execute_operation(OperationSpec(name=name, arguments=arguments)), dict)


def test_count_array_source_and_exclusions_are_explicit() -> None:
    evidence = [EvidenceItem(value=["Al", "", "Mo", "Ti", "Al"])]
    operation = OperationSpec(name="count", arguments={
        "source": {"evidence_index": 0}, "exclude_values": ["", "Ti"],
    })
    validate_operation_grounding(operation, evidence)
    assert execute_operation(operation, evidence) == 3  # 不擅自去重。
    assert execute_operation(OperationSpec(name="count", arguments={
        "values": [{"evidence_index": 0}],
    }), evidence) == 5  # 单一数组只剥一层，不递归拍平。
    assert execute_operation(OperationSpec(name="lookup", arguments={"value": 7})) == 7


@pytest.mark.parametrize("arguments", [{"source": 7}, {"values": [], "source": []}, {}])
def test_count_rejects_ambiguous_collection_contract(arguments: dict) -> None:
    with pytest.raises(OperationExecutionError):
        execute_operation(OperationSpec(name="count", arguments=arguments))


@pytest.mark.parametrize("raw", [6, "6%", "6.0"])
def test_percentage_conversion_is_explicit_and_decimal(raw: object) -> None:
    evidence = [EvidenceItem(value=12), EvidenceItem(value=raw, unit="%")]
    operation = OperationSpec(name="divide", arguments={
        "a": {"evidence_index": 0}, "b": {"evidence_index": 1}, "b_unit": "percent",
    })
    validate_operation_grounding(operation, evidence)
    assert execute_operation(operation, evidence) == Decimal(200)
    assert execute_operation(OperationSpec(name="divide", arguments={"a": 12, "b": 6})) == 2
    assert execute_operation(OperationSpec(name="divide", arguments={
        "a": 12, "b": "0.06", "b_unit": "number",
    })) == 200


def test_percentage_unit_is_inferred_from_referenced_evidence() -> None:
    evidence = [EvidenceItem(value=8.8, unit=None), EvidenceItem(value=8, value_raw="8%", unit="%")]
    operation = OperationSpec(name="divide", arguments={
        "a": {"evidence_index": 0}, "b": {"evidence_index": 1},
    })
    assert execute_operation(operation, evidence) == Decimal(110)


def test_native_selection_works_in_pipeline() -> None:
    operation = OperationSpec.model_validate({"name": "pipeline", "steps": [
        {"id": "name", "name": "concat", "arguments": {
            "values": ["乙校", "二班"], "separator": " / ",
        }},
        {"id": "winner", "name": "argmax", "arguments": {
            "records": [{"label": {"step_id": "name"}, "value": 18}],
            "return_field": "label",
        }},
    ]})
    assert execute_operation(operation) == "乙校 / 二班"
    assert operation_selects_answer(operation)


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("specialist", ["extract", "visual_attribute", "compute", "structure"])
def test_prompts_share_one_answer_boundary(specialist: str, force: bool) -> None:
    prompt = build_specialist_system_prompt(specialist, force_answer=force)
    assert "必须绑定 Evidence" not in prompt
    assert "multi_field 必须使用 Evidence + operation" in prompt
    assert "不要使用 source 参数" not in prompt
    question = QuestionRecord(id=1, file_name="a.png", question_type="extract",
                              question="依次读取两个字段", answer_format="json_array")
    user_prompt = build_question_prompt(question, force_answer=force)
    assert "必须绑定 Evidence" not in user_prompt
    if force:
        assert FORCE_ANSWER_SYSTEM_APPENDIX in user_prompt
    assert "只修复 operation、answer_projection、output" in OPERATION_REPAIR_PROMPT
