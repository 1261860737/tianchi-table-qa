"""语义操作的确定性执行器。"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import time
from decimal import Decimal, InvalidOperation
from typing import Any

from table_qa_agent.schemas import EvidenceItem, OperationSpec


class OperationExecutionError(ValueError):
    """操作名称、参数或数值不合法。"""


class ArgumentGroundingError(ValueError):
    """Operation 参数没有对应到 Evidence 中的可见值。"""


EVIDENCE_REFERENCE_FIELDS = {
    "value",
    "value_raw",
    "row_header",
    "column_header",
    "unit",
    "entity",
    "metric",
    "source_text",
}


def _number(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise OperationExecutionError("布尔值不能作为数字")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace("，", "")
        negative = text.startswith("(") and text.endswith(")")
        if negative:
            text = text[1:-1]
        if text.endswith("%"):
            text = text[:-1]
        try:
            parsed = Decimal(text)
        except InvalidOperation as exc:
            raise OperationExecutionError(f"无法解析数字: {value!r}") from exc
        return -parsed if negative else parsed
    raise OperationExecutionError(f"不支持的数字类型: {type(value).__name__}")


def _values(arguments: dict[str, Any]) -> list[Any]:
    values = arguments.get("values")
    if not isinstance(values, list):
        raise OperationExecutionError("操作参数 values 必须是数组")
    return values


def _binary(arguments: dict[str, Any]) -> tuple[Decimal, Decimal]:
    if "a" not in arguments or "b" not in arguments:
        raise OperationExecutionError("二元操作必须包含 a 和 b")
    return _number(arguments["a"]), _number(arguments["b"])


def _divide(a: Decimal, b: Decimal) -> Decimal:
    if b == 0:
        raise OperationExecutionError("除数不能为 0")
    return a / b


def _execute_lookup(arguments: dict[str, Any]) -> Any:
    if "value" not in arguments:
        raise OperationExecutionError("lookup 必须包含 value")
    return arguments["value"]


def _execute_list(arguments: dict[str, Any]) -> list[Any]:
    return _values(arguments)


def _execute_count(arguments: dict[str, Any]) -> int:
    return len(_values(arguments))


def _execute_add(arguments: dict[str, Any]) -> Decimal:
    a, b = _binary(arguments)
    return a + b


def _execute_subtract(arguments: dict[str, Any]) -> Decimal:
    a, b = _binary(arguments)
    return a - b


def _execute_multiply(arguments: dict[str, Any]) -> Decimal:
    a, b = _binary(arguments)
    return a * b


def _execute_divide(arguments: dict[str, Any]) -> Decimal:
    return _divide(*_binary(arguments))


def _execute_ratio(arguments: dict[str, Any]) -> Decimal:
    return _divide(*_binary(arguments))


def _execute_percentage_change(arguments: dict[str, Any]) -> Decimal:
    if "old_value" not in arguments or "new_value" not in arguments:
        raise OperationExecutionError("percentage_change 必须包含 old_value 和 new_value")
    old = _number(arguments["old_value"])
    new = _number(arguments["new_value"])
    return _divide(new - old, old) * Decimal(100)


def _execute_percentage_of_total(arguments: dict[str, Any]) -> Decimal:
    if "part" not in arguments or "total" not in arguments:
        raise OperationExecutionError("percentage_of_total 必须包含 part 和 total")
    return _divide(_number(arguments["part"]), _number(arguments["total"])) * Decimal(100)


def _execute_percentage_point_difference(arguments: dict[str, Any]) -> Decimal:
    if "old_value" not in arguments or "new_value" not in arguments:
        raise OperationExecutionError(
            "percentage_point_difference 必须包含 old_value 和 new_value"
        )
    return _number(arguments["new_value"]) - _number(arguments["old_value"])


def _numeric_values(arguments: dict[str, Any]) -> list[Decimal]:
    raw_values = _values(arguments)
    null_policy = str(arguments.get("null_policy", "error"))
    if null_policy not in {"error", "ignore", "zero"}:
        raise OperationExecutionError("null_policy 必须是 error/ignore/zero")
    values: list[Decimal] = []
    for value in raw_values:
        if value is None or (isinstance(value, str) and not value.strip()):
            if null_policy == "ignore":
                continue
            if null_policy == "zero":
                values.append(Decimal(0))
                continue
            raise OperationExecutionError("数值数组包含空值")
        values.append(_number(value))
    if not values:
        raise OperationExecutionError("数值数组不能为空")
    return values


def _execute_sum(arguments: dict[str, Any]) -> Decimal:
    return sum(_numeric_values(arguments), start=Decimal(0))


def _execute_average(arguments: dict[str, Any]) -> Decimal:
    values = _numeric_values(arguments)
    return sum(values, start=Decimal(0)) / Decimal(len(values))


def _execute_max(arguments: dict[str, Any]) -> Decimal:
    return max(_numeric_values(arguments))


def _execute_min(arguments: dict[str, Any]) -> Decimal:
    return min(_numeric_values(arguments))


_TIME_PATTERN = re.compile(r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?$")


def _time_seconds(value: Any) -> Decimal:
    if isinstance(value, time):
        return Decimal(value.hour * 3600 + value.minute * 60 + value.second)
    if not isinstance(value, str):
        raise OperationExecutionError(f"无法解析时间: {value!r}")
    matched = _TIME_PATTERN.fullmatch(value.strip())
    if matched is None:
        raise OperationExecutionError(f"无法解析时间: {value!r}")
    hour = int(matched.group("hour"))
    minute = int(matched.group("minute"))
    second = int(matched.group("second") or 0)
    if hour > 23 or minute > 59 or second > 59:
        raise OperationExecutionError(f"时间超出范围: {value!r}")
    return Decimal(hour * 3600 + minute * 60 + second)


def _duration_value(start: Any, end: Any, output_unit: str) -> Decimal:
    start_seconds = _time_seconds(start)
    end_seconds = _time_seconds(end)
    if end_seconds < start_seconds:
        end_seconds += Decimal(24 * 3600)
    seconds = end_seconds - start_seconds
    if output_unit == "minute":
        return seconds / Decimal(60)
    if output_unit == "hour":
        return seconds / Decimal(3600)
    if output_unit == "second":
        return seconds
    raise OperationExecutionError("output_unit 必须是 second/minute/hour")


def _execute_duration(arguments: dict[str, Any]) -> Decimal:
    if "start" not in arguments or "end" not in arguments:
        raise OperationExecutionError("duration 必须包含 start 和 end")
    return _duration_value(
        arguments["start"],
        arguments["end"],
        str(arguments.get("output_unit", "minute")),
    )


def _execute_sum_durations(arguments: dict[str, Any]) -> Decimal:
    ranges = arguments.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise OperationExecutionError("sum_durations.ranges 必须是非空数组")
    output_unit = str(arguments.get("output_unit", "minute"))
    total = Decimal(0)
    for index, item in enumerate(ranges):
        if not isinstance(item, dict) or "start" not in item or "end" not in item:
            raise OperationExecutionError(f"ranges[{index}] 必须包含 start 和 end")
        total += _duration_value(item["start"], item["end"], output_unit)
    return total


def _execute_arg_extreme(arguments: dict[str, Any], *, find_max: bool) -> dict[str, Any]:
    records = arguments.get("records")
    if not isinstance(records, list) or not records:
        raise OperationExecutionError("argmax/argmin.records 必须是非空数组")
    value_field = str(arguments.get("value_field", "value"))
    label_field = str(arguments.get("label_field", "label"))
    expanded_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or value_field not in record:
            raise OperationExecutionError(f"records[{index}] 缺少数值字段 {value_field!r}")
        label = record.get(label_field)
        numeric_value = record[value_field]
        if isinstance(label, (list, tuple)) or isinstance(numeric_value, (list, tuple)):
            if not isinstance(label, (list, tuple)) or not isinstance(
                numeric_value, (list, tuple)
            ):
                raise OperationExecutionError(f"records[{index}] 的标签与数值数组必须成对出现")
            if len(label) != len(numeric_value) or not label:
                raise OperationExecutionError(f"records[{index}] 的标签与数值数组长度不一致")
            expanded_records.extend(
                {**record, label_field: item_label, value_field: item_value}
                for item_label, item_value in zip(label, numeric_value, strict=True)
            )
        else:
            expanded_records.append(record)

    normalized: list[tuple[Decimal, dict[str, Any]]] = []
    for index, record in enumerate(expanded_records):
        label = record.get(label_field)
        if isinstance(label, bool) or isinstance(label, (dict, list, tuple)) or label is None:
            raise OperationExecutionError(
                f"records[{index}] 的标签字段 {label_field!r} 必须是非空标量"
            )
        if isinstance(label, str) and not label.strip():
            raise OperationExecutionError(
                f"records[{index}] 的标签字段 {label_field!r} 必须是非空标量"
            )
        normalized.append((_number(record[value_field]), record))
    selector = max if find_max else min
    return selector(normalized, key=lambda item: item[0])[1]


def _execute_argmax(arguments: dict[str, Any]) -> dict[str, Any]:
    return _execute_arg_extreme(arguments, find_max=True)


def _execute_argmin(arguments: dict[str, Any]) -> dict[str, Any]:
    return _execute_arg_extreme(arguments, find_max=False)


def _execute_boolean(arguments: dict[str, Any]) -> Any:
    if "value" not in arguments:
        raise OperationExecutionError("boolean 必须包含 value")
    return arguments["value"]


def _execute_concat(arguments: dict[str, Any]) -> str:
    values = _values(arguments)
    separator = str(arguments.get("separator", ""))
    return separator.join(str(value) for value in values)


OPERATIONS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "lookup": _execute_lookup,
    "list": _execute_list,
    "count": _execute_count,
    "add": _execute_add,
    "subtract": _execute_subtract,
    "multiply": _execute_multiply,
    "divide": _execute_divide,
    "ratio": _execute_ratio,
    "percentage_change": _execute_percentage_change,
    "percentage_of_total": _execute_percentage_of_total,
    "percentage_point_difference": _execute_percentage_point_difference,
    "sum": _execute_sum,
    "average": _execute_average,
    "max": _execute_max,
    "min": _execute_min,
    "duration": _execute_duration,
    "sum_durations": _execute_sum_durations,
    "argmax": _execute_argmax,
    "argmin": _execute_argmin,
    "boolean": _execute_boolean,
    "concat": _execute_concat,
}


def _execute_atomic(name: str, arguments: dict[str, Any]) -> Any:
    if not isinstance(arguments, dict):
        raise OperationExecutionError(
            f"{name} 参数解析后必须是对象，实际为 {type(arguments).__name__}；"
            'lookup 引用应放在 {"value": 引用} 内'
        )
    try:
        executor = OPERATIONS[name]
    except KeyError as exc:
        raise OperationExecutionError(f"未知操作: {name}") from exc
    return executor(arguments)


def _resolve_references(
    value: Any,
    evidence: list[EvidenceItem],
    step_results: dict[str, Any],
) -> Any:
    """递归解析 Evidence 和前序步骤引用。"""

    if isinstance(value, dict):
        if "evidence_id" in value:
            allowed_keys = {"evidence_id", "field"}
            if set(value) - allowed_keys:
                raise OperationExecutionError("Evidence 引用只能包含 evidence_id 和 field")
            evidence_id = value["evidence_id"]
            matches = [item for item in evidence if item.id == evidence_id]
            if len(matches) != 1:
                raise OperationExecutionError(f"evidence_id 不存在或不唯一: {evidence_id!r}")
            field = value.get("field", "value")
            if field not in EVIDENCE_REFERENCE_FIELDS:
                raise OperationExecutionError(f"不支持的 Evidence 字段: {field}")
            return getattr(matches[0], field)
        if "evidence_index" in value:
            allowed_keys = {"evidence_index", "field"}
            if set(value) - allowed_keys:
                raise OperationExecutionError("Evidence 引用只能包含 evidence_index 和 field")
            index = value["evidence_index"]
            if not isinstance(index, int) or isinstance(index, bool):
                raise OperationExecutionError("evidence_index 必须是整数")
            if not 0 <= index < len(evidence):
                raise OperationExecutionError(f"evidence_index 越界: {index}")
            field = value.get("field", "value")
            if field not in EVIDENCE_REFERENCE_FIELDS:
                raise OperationExecutionError(f"不支持的 Evidence 字段: {field}")
            return getattr(evidence[index], field)
        if "step_id" in value:
            allowed_keys = {"step_id", "field", "index"}
            if set(value) - allowed_keys or ("field" in value and "index" in value):
                raise OperationExecutionError("步骤引用只能包含 step_id 以及 field 或 index")
            step_id = value["step_id"]
            if not isinstance(step_id, str) or step_id not in step_results:
                raise OperationExecutionError(f"步骤尚未执行或不存在: {step_id!r}")
            result = step_results[step_id]
            if "field" in value:
                field = value["field"]
                if (
                    not isinstance(field, str)
                    or not isinstance(result, dict)
                    or field not in result
                ):
                    raise OperationExecutionError(f"步骤 {step_id!r} 不包含字段: {field!r}")
                return result[field]
            if "index" in value:
                index = value["index"]
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or not isinstance(result, (list, tuple))
                    or not -len(result) <= index < len(result)
                ):
                    raise OperationExecutionError(f"步骤 {step_id!r} 无法读取索引: {index!r}")
                return result[index]
            return result
        return {
            key: _resolve_references(item, evidence, step_results) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_references(item, evidence, step_results) for item in value]
    return value


def execute_operation(
    operation: OperationSpec,
    evidence: list[EvidenceItem] | None = None,
) -> Any:
    """解析取证引用并执行白名单操作，不运行模型生成的代码。"""

    evidence_items = evidence or []
    if operation.name != "pipeline":
        arguments = _resolve_references(operation.arguments, evidence_items, {})
        return _execute_atomic(operation.name, arguments)

    step_results: dict[str, Any] = {}
    for step in operation.steps:
        arguments = _resolve_references(step.arguments, evidence_items, step_results)
        step_results[step.id] = _execute_atomic(step.name, arguments)
    return step_results[operation.steps[-1].id]


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, dict):
        flattened: list[Any] = []
        for item in value.values():
            flattened.extend(_flatten(item))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]


_CONFIG_ARGUMENT_KEYS = {
    "output_unit",
    "value_field",
    "label_field",
    "null_policy",
    "separator",
}


def _direct_argument_values(value: Any) -> list[Any]:
    """提取参数中的字面量；Evidence/步骤引用交由解析器验证。"""

    if isinstance(value, dict):
        if "evidence_index" in value or "evidence_id" in value or "step_id" in value:
            return []
        flattened: list[Any] = []
        for key, item in value.items():
            if key in _CONFIG_ARGUMENT_KEYS:
                continue
            flattened.extend(_direct_argument_values(item))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_direct_argument_values(item))
        return flattened
    return [value]


def _canonical(value: Any) -> tuple[str, str]:
    if isinstance(value, bool) or value is None:
        return type(value).__name__, str(value)
    try:
        return "number", str(_number(value).normalize())
    except OperationExecutionError:
        return "text", str(value).strip()


def validate_operation_grounding(operation: OperationSpec, evidence: list[EvidenceItem]) -> None:
    """确保操作使用的是 Evidence 中的值，而不是模型暗算出的结果。"""

    if operation.name == "boolean":
        # 判断结论本身是从证据派生出的文本，不要求在表格中逐字出现。
        return

    evidence_values = {
        _canonical(value)
        for item in evidence
        for field in (
            item.value,
            item.value_raw,
            item.row_header,
            item.column_header,
            item.unit,
            item.entity,
            item.metric,
            item.source_text,
        )
        for value in _flatten(field)
        if value is not None
    }
    argument_sources = [operation.arguments] if operation.name != "pipeline" else [
        step.arguments for step in operation.steps if step.name != "boolean"
    ]
    argument_values = {
        _canonical(value)
        for arguments in argument_sources
        for value in _direct_argument_values(arguments)
        if value is not None
    }
    missing = argument_values - evidence_values
    if missing:
        rendered = ", ".join(repr(value) for _, value in sorted(missing))
        raise ArgumentGroundingError(f"Operation 参数未在 Evidence 中出现: {rendered}")
