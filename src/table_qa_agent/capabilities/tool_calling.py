"""Agent 可请求的共享工具定义及参数校验。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from table_qa_agent.client import ModelToolCall


class InspectTableArguments(BaseModel):
    """请求定位、高清裁剪并 OCR 与问题有关的表格区域。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=500)


AGENT_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_table_region",
            "description": (
                "当原页面中的表格文字太小、行列对应关系不清楚、计算输入缺失或候选冲突时，"
                "定位相关表格区域，生成高清裁剪和 OCR 文本后再继续回答。内容已经足够清晰时"
                "不要调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要在表格中读取的字段、行列或计算输入。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "当前页面为什么不足以可靠回答。",
                    },
                },
                "required": ["query", "reason"],
                "additionalProperties": False,
            },
        },
    }
]


def validate_tool_call(call: ModelToolCall) -> InspectTableArguments:
    """只允许已注册工具，并在执行任何本地能力前严格校验参数。"""

    if call.name != "inspect_table_region":
        raise ValueError(f"未注册的 Agent 工具: {call.name}")
    return InspectTableArguments.model_validate(call.arguments)
