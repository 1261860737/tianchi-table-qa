from decimal import Decimal

import pytest

from table_qa_agent.answer import project_answer
from table_qa_agent.executor import (
    ArgumentGroundingError,
    OperationExecutionError,
    execute_operation,
    validate_operation_grounding,
)
from table_qa_agent.normalizer import normalize_answer
from table_qa_agent.schemas import AnswerProjection, EvidenceItem, OperationSpec, OperationStep


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("lookup", {"value": "OpenAI"}, "OpenAI"),
        ("list", {"values": [1, "a"]}, [1, "a"]),
        ("count", {"values": ["a", "b", "c"]}, 3),
        ("add", {"a": "1,200", "b": 30}, Decimal("1230")),
        ("subtract", {"a": 150, "b": 120}, Decimal("30")),
        ("multiply", {"a": 2.5, "b": 4}, Decimal("10.0")),
        ("divide", {"a": 9, "b": 4}, Decimal("2.25")),
        ("ratio", {"a": 1, "b": 4}, Decimal("0.25")),
        (
            "percentage_change",
            {"old_value": 120, "new_value": 150},
            Decimal("25.00"),
        ),
        (
            "percentage_of_total",
            {"part": 5, "total": 20},
            Decimal("25.00"),
        ),
        ("sum", {"values": [1, 2, 3]}, Decimal("6")),
        ("average", {"values": [1, 2, 3]}, Decimal("2")),
        ("max", {"values": [1, 8, 3]}, Decimal("8")),
        ("min", {"values": [1, 8, 3]}, Decimal("1")),
        ("concat", {"values": ["Sandeep", "Singh"], "separator": " "}, "Sandeep Singh"),
        (
            "percentage_point_difference",
            {"old_value": "21.5%", "new_value": "22.6%"},
            Decimal("1.1"),
        ),
        (
            "duration",
            {"start": "12:00", "end": "14:00", "output_unit": "minute"},
            Decimal("120"),
        ),
    ],
)
def test_operations(name: str, arguments: dict, expected: object) -> None:
    assert execute_operation(OperationSpec(name=name, arguments=arguments)) == expected


def test_divide_by_zero_is_rejected() -> None:
    with pytest.raises(OperationExecutionError, match="除数不能为 0"):
        execute_operation(OperationSpec(name="divide", arguments={"a": 1, "b": 0}))


def test_operation_arguments_must_come_from_evidence() -> None:
    evidence = [
        EvidenceItem(value=120, value_raw="120"),
        EvidenceItem(value=150, value_raw="150"),
    ]
    validate_operation_grounding(
        OperationSpec(name="subtract", arguments={"a": 150, "b": 120}), evidence
    )
    with pytest.raises(ArgumentGroundingError):
        validate_operation_grounding(
            OperationSpec(name="subtract", arguments={"a": 160, "b": 120}),
            evidence,
        )


def test_operation_can_reference_evidence() -> None:
    evidence = [
        EvidenceItem(value=120, value_raw="120"),
        EvidenceItem(value=150, value_raw="150"),
    ]
    operation = OperationSpec(
        name="subtract",
        arguments={"a": {"evidence_index": 1}, "b": {"evidence_index": 0}},
    )
    validate_operation_grounding(operation, evidence)
    assert execute_operation(operation, evidence) == Decimal("30")


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("value", 42),
        ("value_raw", "42.0"),
        ("row_header", "华东"),
        ("column_header", "营业收入"),
        ("unit", "万元"),
        ("entity", "甲公司"),
        ("metric", "本期金额"),
        ("source_text", "甲公司本期金额 42.0 万元"),
    ],
)
@pytest.mark.parametrize("reference_key", ["evidence_id", "evidence_index"])
def test_all_semantic_evidence_fields_support_id_and_index_references(
    field: str,
    expected: object,
    reference_key: str,
) -> None:
    evidence = [
        EvidenceItem(
            id="cell_1",
            entity="甲公司",
            metric="本期金额",
            row_header="华东",
            column_header="营业收入",
            value_raw="42.0",
            value=42,
            unit="万元",
            source_text="甲公司本期金额 42.0 万元",
        )
    ]
    reference_value: str | int = "cell_1" if reference_key == "evidence_id" else 0
    operation = OperationSpec(
        name="lookup",
        arguments={"value": {reference_key: reference_value, "field": field}},
    )

    validate_operation_grounding(operation, evidence)
    assert execute_operation(operation, evidence) == expected


def test_pipeline_uses_only_evidence_and_prior_step_results() -> None:
    evidence = [
        EvidenceItem(value=10529762.03, value_raw="10,529,762.03"),
        EvidenceItem(value=5264881.02, value_raw="5,264,881.02"),
        EvidenceItem(value=1987509.8, value_raw="1,987,509.80"),
        EvidenceItem(value=1987509.8, value_raw="1,987,509.80"),
    ]
    operation = OperationSpec(
        name="pipeline",
        steps=[
            OperationStep(
                id="part_sum",
                name="sum",
                arguments={"values": [{"evidence_index": 1}, {"evidence_index": 3}]},
            ),
            OperationStep(
                id="total_sum",
                name="sum",
                arguments={"values": [{"evidence_index": 0}, {"evidence_index": 2}]},
            ),
            OperationStep(
                id="answer",
                name="percentage_of_total",
                arguments={
                    "part": {"step_id": "part_sum"},
                    "total": {"step_id": "total_sum"},
                },
            ),
        ],
    )
    validate_operation_grounding(operation, evidence)
    result = execute_operation(operation, evidence)
    assert normalize_answer(result, "number", {"decimals": 2}) == "57.94"


def test_structure_operation_can_use_evidence_metadata() -> None:
    evidence = [
        EvidenceItem(
            column_header="单位名称",
            row_header="1",
            value="杭州海兴电力科技股份有限公司",
        )
    ]
    operation = OperationSpec(
        name="lookup",
        arguments={
            "value": {
                "columns": ["单位名称"],
                "rows": [["1", "杭州海兴电力科技股份有限公司"]],
            }
        },
    )
    validate_operation_grounding(operation, evidence)


def test_duration_supports_cross_midnight() -> None:
    operation = OperationSpec(
        name="duration",
        arguments={"start": "23:30", "end": "00:15", "output_unit": "minute"},
    )

    assert execute_operation(operation) == Decimal("45")


def test_average_null_policy_is_explicit() -> None:
    operation = OperationSpec(
        name="average",
        arguments={"values": [2, None, 4], "null_policy": "ignore"},
    )

    assert execute_operation(operation) == Decimal("3")


def test_argmax_returns_complete_record_then_projects_label() -> None:
    operation = OperationSpec(
        name="argmax",
        arguments={
            "records": [
                {"label": "公司A", "value": 320000},
                {"label": "公司B", "value": 510000},
            ],
            "value_field": "value",
        },
    )
    result = execute_operation(operation)

    assert result == {"label": "公司B", "value": 510000}
    assert project_answer(result, AnswerProjection(mode="path", path=["label"])) == "公司B"


def test_argmax_accepts_numeric_label_when_question_requests_numeric_identifier() -> None:
    operation = OperationSpec(
        name="argmax",
        arguments={"records": [{"label": 510000, "value": 510000}]},
    )

    assert execute_operation(operation) == {"label": 510000, "value": 510000}


def test_argmax_can_reference_evidence_entity_by_index() -> None:
    evidence = [
        EvidenceItem(entity="房屋及建筑物", value=161545571.17),
        EvidenceItem(entity="管网设备", value=429265075.91),
        EvidenceItem(entity="机器设备", value=224837402.61),
    ]
    operation = OperationSpec(
        name="argmax",
        arguments={
            "records": [
                {
                    "label": {"evidence_index": index, "field": "entity"},
                    "value": {"evidence_index": index, "field": "value"},
                }
                for index in range(3)
            ]
        },
    )

    validate_operation_grounding(operation, evidence)
    result = execute_operation(operation, evidence)

    assert result == {"label": "管网设备", "value": 429265075.91}


def test_argmax_expands_aligned_label_and_value_arrays() -> None:
    operation = OperationSpec(
        name="argmax",
        arguments={
            "records": [
                {"label": ["科目 A", "科目 B", "科目 C"], "value": [10, 30, 20]}
            ]
        },
    )

    assert execute_operation(operation) == {"label": "科目 B", "value": 30}


def test_pipeline_step_result_can_project_field_and_index() -> None:
    operation = OperationSpec(
        name="pipeline",
        steps=[
            OperationStep(
                id="record",
                name="argmin",
                arguments={
                    "records": [
                        {"label": "甲", "value": 3},
                        {"label": "乙", "value": 1},
                    ]
                },
            ),
            OperationStep(
                id="items",
                name="list",
                arguments={
                    "values": [
                        {"step_id": "record", "field": "value"},
                        {"step_id": "record", "field": "label"},
                    ]
                },
            ),
            OperationStep(
                id="answer",
                name="lookup",
                arguments={"value": {"step_id": "items", "index": 1}},
            ),
        ],
    )

    assert execute_operation(operation) == "乙"


def test_operation_can_reference_stable_evidence_id() -> None:
    evidence = [EvidenceItem(id="rent_b", value=510000, value_raw="510,000")]
    operation = OperationSpec(
        name="lookup",
        arguments={"value": {"evidence_id": "rent_b"}},
    )

    validate_operation_grounding(operation, evidence)
    assert execute_operation(operation, evidence) == 510000
def test_root_reference_reports_operation_error_instead_of_type_error():
    from table_qa_agent.executor import OperationExecutionError, execute_operation
    from table_qa_agent.schemas import EvidenceItem, OperationSpec

    with pytest.raises(OperationExecutionError, match="参数解析后必须是对象"):
        execute_operation(
            OperationSpec(name="lookup", arguments={"evidence_index": 0}),
            [EvidenceItem(value=13470966.56)],
        )
