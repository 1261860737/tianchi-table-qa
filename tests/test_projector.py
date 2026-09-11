import pytest

from table_qa_agent.answer import AnswerProjectionError, project_answer
from table_qa_agent.schemas import AnswerProjection


def test_projection_selects_argmax_label() -> None:
    result = {"label": "公司B", "value": 510000}
    projection = AnswerProjection(mode="path", path=["label"])

    assert project_answer(result, projection) == "公司B"


def test_projection_preserves_requested_field_order() -> None:
    result = {"name": "张三", "score": 92}
    projection = AnswerProjection(mode="fields", fields=["score", "name"])

    assert project_answer(result, projection) == [92, "张三"]


def test_projection_rejects_missing_field() -> None:
    with pytest.raises(AnswerProjectionError, match="缺少投影字段"):
        project_answer({"value": 1}, AnswerProjection(mode="path", path=["label"]))
