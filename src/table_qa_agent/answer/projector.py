"""从完整工具结果中选择题目要求的最终字段。"""

from __future__ import annotations

from typing import Any

from table_qa_agent.schemas import AnswerProjection


class AnswerProjectionError(ValueError):
    """工具结果无法按 TaskPlan 投影。"""


def _path_value(value: Any, path: list[str | int]) -> Any:
    current = value
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, (list, tuple)):
                raise AnswerProjectionError(f"无法从 {type(current).__name__} 读取索引 {part}")
            try:
                current = current[part]
            except IndexError as exc:
                raise AnswerProjectionError(f"投影索引越界: {part}") from exc
            continue
        if not isinstance(current, dict) or part not in current:
            raise AnswerProjectionError(f"工具结果缺少投影字段: {part}")
        current = current[part]
    return current


def project_answer(value: Any, projection: AnswerProjection | None) -> Any:
    if projection is None or projection.mode == "identity":
        return value
    if projection.mode == "path":
        return _path_value(value, projection.path)
    if not isinstance(value, dict):
        raise AnswerProjectionError("fields 投影要求工具结果是对象")
    missing = [field for field in projection.fields if field not in value]
    if missing:
        raise AnswerProjectionError(f"工具结果缺少投影字段: {missing}")
    return [value[field] for field in projection.fields]
