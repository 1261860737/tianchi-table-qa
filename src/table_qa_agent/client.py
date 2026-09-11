"""OpenAI 兼容视觉模型客户端。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from table_qa_agent.config import ProviderConfig
from table_qa_agent.schemas import TokenUsage


class ModelConfigurationError(RuntimeError):
    """模型连接配置缺失。"""


@dataclass(slots=True)
class ModelToolCall:
    """模型通过 OpenAI 兼容协议请求执行的函数。"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ModelCompletion:
    text: str
    usage: TokenUsage
    tool_calls: list[ModelToolCall] = field(default_factory=list)


class OpenAICompatibleVLClient:
    """支持多图 Base64 Data URL 的同步 Chat Completions 客户端。"""

    def __init__(self, config: ProviderConfig) -> None:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ModelConfigurationError(f"未设置环境变量 {config.api_key_env}，无法调用模型")
        self.config = config
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    def complete(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_content: list[dict[str, object]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> ModelCompletion:
        content = self._with_pixel_hints(image_content)
        content.append({"type": "text", "text": user_text})
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        # 部分 OpenAI 兼容服务不允许 response_format 与 tools 同时出现；最终文本仍由
        # Prompt 和 json-repair 约束为 JSON。
        if self.config.json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
            kwargs["parallel_tool_calls"] = False
        kwargs["extra_body"] = {"enable_thinking": self.config.enable_thinking}

        retrying = Retrying(
            stop=stop_after_attempt(self.config.max_retries + 1),
            # 并发限流时加入随机抖动，避免多个请求同时重试。
            wait=wait_random_exponential(multiplier=1, max=20),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        response = retrying(self.client.chat.completions.create, **kwargs)
        if not response.choices:
            raise RuntimeError("模型响应没有 choices")
        message = response.choices[0].message
        text = message.content if isinstance(message.content, str) else ""
        tool_calls: list[ModelToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"工具 {call.function.name} 的 arguments 不是合法 JSON"
                ) from exc
            if not isinstance(arguments, dict):
                raise RuntimeError(f"工具 {call.function.name} 的 arguments 必须是对象")
            tool_calls.append(
                ModelToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )
        if not text.strip() and not tool_calls:
            raise RuntimeError("模型既没有返回文本，也没有请求工具")

        usage = response.usage
        return ModelCompletion(
            text=text,
            usage=TokenUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            ),
            tool_calls=tool_calls,
        )

    def _with_pixel_hints(self, content: list[dict[str, object]]) -> list[dict[str, object]]:
        enriched: list[dict[str, object]] = []
        for block in content:
            copied = dict(block)
            if copied.get("type") == "image_url":
                if self.config.min_pixels is not None:
                    copied["min_pixels"] = self.config.min_pixels
                if self.config.max_pixels is not None:
                    copied["max_pixels"] = self.config.max_pixels
            enriched.append(copied)
        return enriched
