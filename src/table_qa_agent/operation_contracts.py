"""已解析函数参数的形状合同；数值语义仍交由执行器处理。"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Arguments(BaseModel):
    # 兼容旧日志的辅助字段，不做隐式类型转换。
    model_config = ConfigDict(strict=True, extra="allow")


class ValueArguments(Arguments):
    value: Any


class ValuesArguments(Arguments):
    values: list[Any]


class ListArguments(ValuesArguments):
    select_field: str | None = None


class CountArguments(Arguments):
    values: list[Any] | None = None
    source: list[Any] | None = None
    exclude_values: list[Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_collection(self) -> "CountArguments":
        if (self.values is None) == (self.source is None):
            raise ValueError("count 必须且只能提供 values 或 source 数组")
        return self


class NumericValuesArguments(ValuesArguments):
    null_policy: Literal["error", "ignore", "zero"] = "error"


class BinaryArguments(Arguments):
    a: Any
    b: Any
    a_unit: Literal["number", "percent"] = "number"
    b_unit: Literal["number", "percent"] = "number"


class ChangeArguments(Arguments):
    old_value: Any
    new_value: Any


class TotalArguments(Arguments):
    part: Any
    total: Any


class TimeRange(Arguments):
    start: Any
    end: Any


class DurationArguments(TimeRange):
    output_unit: Literal["second", "minute", "hour"] = "minute"


class RangesArguments(Arguments):
    ranges: list[TimeRange] = Field(min_length=1)
    output_unit: Literal["second", "minute", "hour"] = "minute"


class ExtremeArguments(Arguments):
    records: list[dict[str, Any]] = Field(min_length=1)
    value_field: str = "value"
    label_field: str = "label"
    return_field: str | None = None


class ConcatArguments(ValuesArguments):
    separator: str = ""


ARGUMENT_MODELS: dict[str, type[Arguments]] = {
    "lookup": ValueArguments,
    "boolean": ValueArguments,
    "list": ListArguments,
    "count": CountArguments,
    **dict.fromkeys(("add", "subtract", "multiply", "divide", "ratio"), BinaryArguments),
    **dict.fromkeys(("percentage_change", "percentage_point_difference"), ChangeArguments),
    "percentage_of_total": TotalArguments,
    **dict.fromkeys(("sum", "average", "max", "min"), NumericValuesArguments),
    "duration": DurationArguments,
    "sum_durations": RangesArguments,
    "argmax": ExtremeArguments,
    "argmin": ExtremeArguments,
    "concat": ConcatArguments,
}


def validate_arguments(name: str, arguments: dict[str, Any]) -> None:
    """仅验证、不改写输入，避免影响历史合法操作的值和顺序。"""
    ARGUMENT_MODELS[name].model_validate(arguments)
