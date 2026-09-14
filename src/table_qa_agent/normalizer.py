"""根据比赛 answer_format 生成稳定、可比较的答案字符串。"""

from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from table_qa_agent.schemas import (
    AnswerFormat,
    QuestionRecord,
    TableStructureAnswer,
)


class AnswerNormalizationError(ValueError):
    """工具结果与题目要求的输出格式不兼容。"""


def _as_decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise AnswerNormalizationError("布尔值不是 number")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace("，", "")
        if text.endswith("%"):
            text = text[:-1]
        try:
            return Decimal(text)
        except InvalidOperation as exc:
            raise AnswerNormalizationError(f"无法转成 number: {value!r}") from exc
    raise AnswerNormalizationError(f"无法转成 number: {type(value).__name__}")


def _format_decimal(value: Decimal, decimals: int | None = None) -> str:
    if not value.is_finite():
        raise AnswerNormalizationError("答案不能是 NaN 或 Infinity")
    if decimals is not None:
        if decimals < 0 or decimals > 12:
            raise AnswerNormalizationError("decimals 必须在 0 到 12 之间")
        quantum = Decimal(1).scaleb(-decimals)
        value = value.quantize(quantum, rounding=ROUND_HALF_UP)
        return f"{value:.{decimals}f}"
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f").rstrip("0").rstrip(".")


def _jsonable(value: Any) -> Any:
    if value is None:
        # 比赛规定：数组中的空值必须填写空字符串，不能输出 null。
        return ""
    if isinstance(value, Decimal):
        text = _format_decimal(value)
        return int(text) if "." not in text else float(text)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parse_json(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON 不允许 {value}")

    try:
        return json.loads(text, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise AnswerNormalizationError("答案不是合法 JSON") from exc


def _contains_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_null(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_null(item) for item in value)
    return False


def _unwrap_singleton_scalar(value: Any, answer_format: AnswerFormat) -> Any:
    """收紧标量接口：仅解包无歧义的单元素序列。

    模型或确定性 Operation 有时会把单个答案按 ``[value]`` 返回。题目已经用
    ``answer_format`` 声明了标量语义，因此 string/number 下的一层单元素
    list/tuple 只是接口形状噪声，可以在统一规范化边界安全消除。多元素序列
    和对象仍交给各格式原有逻辑处理，避免猜测应保留哪个值。
    """

    if answer_format in {"string", "number"} and isinstance(value, (list, tuple)):
        if len(value) == 1:
            return value[0]
    return value


EXPLANATORY_PREFIX = re.compile(
    r"^(?:根据(?:表格|图片|文档)|由(?:表格|图片|文档)|从(?:表格|图片|文档)|答案(?:是|为)|可知)"
)


def validate_answer_text(
    answer: object,
    question: QuestionRecord,
    *,
    allow_blank: bool = True,
) -> list[str]:
    """按题目元数据检查最终 answer，返回可读的违规原因。"""

    text = "" if answer is None else str(answer).strip()
    if not text:
        return [] if allow_blank else ["答案为空"]
    if text.casefold() in {"none", "null"}:
        return ["无法作答时必须填写空字符串，不能填写 None/null"]
    if EXPLANATORY_PREFIX.search(text):
        return ["答案包含说明性前缀"]

    if question.answer_format == "number":
        if "," in text or "，" in text:
            return ["数字答案不能包含千分位逗号"]
        try:
            _as_decimal(text)
        except AnswerNormalizationError as exc:
            return [str(exc)]
        return []

    if question.answer_format == "json_array":
        try:
            value = _parse_json(text)
        except AnswerNormalizationError as exc:
            return [str(exc)]
        errors: list[str] = []
        if not isinstance(value, list):
            errors.append("json_array 答案必须是 JSON 数组")
        if _contains_null(value):
            errors.append('JSON 数组中的空值必须使用空字符串 ""，不能使用 null')
        return errors

    if question.answer_format == "json":
        try:
            value = _parse_json(text)
            TableStructureAnswer.model_validate(value)
        except AnswerNormalizationError as exc:
            return [str(exc)]
        except ValidationError as exc:
            details = []
            errors = exc.errors(include_url=False)
            for error in errors[:5]:
                location = ".".join(str(item) for item in error["loc"]) or "root"
                details.append(f"{location}: {error['msg']}")
            if len(errors) > 5:
                details.append(f"另有 {len(errors) - 5} 个错误")
            return ["结构恢复答案不符合 row_count/col_count/cells 协议: " + "; ".join(details)]
        return []

    return []


def normalize_answer(
    value: Any,
    answer_format: AnswerFormat,
    output_hints: dict[str, Any] | None = None,
) -> str:
    """标准化工具结果；所有提交答案最终都写成字符串。"""

    value = _unwrap_singleton_scalar(value, answer_format)
    hints = output_hints or {}
    decimals_raw = hints.get("decimals")
    decimals = int(decimals_raw) if decimals_raw is not None else None
    suffix = str(hints.get("suffix") or "")

    if answer_format == "number":
        if isinstance(value, (list, tuple, dict)):
            raise AnswerNormalizationError("number 不能接收数组或对象")
        return _format_decimal(_as_decimal(value), decimals)

    if answer_format == "string":
        if value is None:
            return ""
        if isinstance(value, bool):
            text = "是" if value else "否"
        elif isinstance(value, Decimal):
            text = _format_decimal(value, decimals)
        elif isinstance(value, (list, tuple, dict)):
            text = json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))
        else:
            text = str(value).strip()
        return f"{text}{suffix}"

    if answer_format == "json_array":
        if not isinstance(value, (list, tuple)):
            value = [value]
        return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))

    if answer_format == "json":
        try:
            structure = TableStructureAnswer.model_validate(_jsonable(value))
        except ValidationError as exc:
            raise AnswerNormalizationError(
                "结构恢复答案必须包含合法的 row_count、col_count 和 cells"
            ) from exc
        return json.dumps(
            structure.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    raise AnswerNormalizationError(f"未知 answer_format: {answer_format}")
