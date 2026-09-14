from decimal import Decimal

import pytest

from table_qa_agent.normalizer import (
    AnswerNormalizationError,
    normalize_answer,
    validate_answer_text,
)
from table_qa_agent.schemas import QuestionRecord


def test_number_removes_redundant_zeroes() -> None:
    assert normalize_answer(Decimal("25.000"), "number") == "25"


def test_string_percentage_respects_rounding_hint() -> None:
    assert normalize_answer(Decimal("12.345"), "string", {"decimals": 1, "suffix": "%"}) == "12.3%"


@pytest.mark.parametrize("container", [["迭代与表达"], ("迭代与表达",)])
def test_string_unwraps_singleton_sequence_at_scalar_boundary(container: object) -> None:
    assert normalize_answer(container, "string") == "迭代与表达"


@pytest.mark.parametrize("container", [[Decimal("25.0")], (Decimal("25.0"),)])
def test_number_unwraps_singleton_sequence_at_scalar_boundary(container: object) -> None:
    assert normalize_answer(container, "number") == "25"


def test_string_keeps_multi_value_sequence_without_guessing() -> None:
    assert normalize_answer(["TA6", "4.0~5.5"], "string") == '["TA6","4.0~5.5"]'


def test_json_array_is_compact_and_keeps_chinese() -> None:
    assert normalize_answer([Decimal("1"), "华东"], "json_array") == '[1,"华东"]'


def test_json_array_wraps_scalar_for_dirty_contract() -> None:
    assert normalize_answer(Decimal("135088205.15"), "json_array") == "[135088205.15]"


def test_json_array_uses_empty_strings_instead_of_null() -> None:
    assert normalize_answer([1, None, {"值": None}], "json_array") == '[1,"",{"值":""}]'


def test_none_string_answer_becomes_blank() -> None:
    assert normalize_answer(None, "string") == ""


def test_structure_answer_uses_official_contract() -> None:
    value = {
        "row_count": 4,
        "col_count": 3,
        "cells": [
            {"text": "项目", "row": 0, "col": 0, "rowspan": 2, "colspan": 1},
            {"text": "金额", "row": 0, "col": 1, "rowspan": 1, "colspan": 2},
        ],
    }
    assert normalize_answer(value, "json") == (
        '{"row_count":4,"col_count":3,"cells":['
        '{"text":"项目","row":0,"col":0,"rowspan":2,"colspan":1},'
        '{"text":"金额","row":0,"col":1,"rowspan":1,"colspan":2}]}'
    )


@pytest.mark.parametrize(
    "value",
    [
        ["项目", "金额"],
        {"headers": ["项目", "金额"]},
        {
            "row_count": 1,
            "col_count": 1,
            "cells": [{"text": "项目", "row": 0, "col": 0, "rowspan": 1, "colspan": 2}],
        },
        {
            "row_count": 2,
            "col_count": 2,
            "cells": [
                {"text": "合并", "row": 0, "col": 0, "rowspan": 1, "colspan": 2},
                {"text": "重复", "row": 0, "col": 1, "rowspan": 1, "colspan": 1},
            ],
        },
    ],
)
def test_structure_answer_rejects_invalid_shapes(value: object) -> None:
    with pytest.raises(AnswerNormalizationError):
        normalize_answer(value, "json")


def test_answer_contract_rejects_none_and_explanatory_prefix() -> None:
    question = QuestionRecord(
        id=1,
        file_name="001.png",
        question_type="extract",
        question="值是多少？",
        answer_format="string",
    )
    assert validate_answer_text("None", question)
    assert validate_answer_text("根据表格可知，答案为 1", question)
    assert validate_answer_text("", question) == []
    assert validate_answer_text("", question, allow_blank=False) == ["答案为空"]


def test_number_rejects_multi_value_list() -> None:
    with pytest.raises(AnswerNormalizationError):
        normalize_answer([1, 2], "number")
