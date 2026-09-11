from types import SimpleNamespace

from table_qa_agent.capabilities import AGENT_TOOL_DEFINITIONS
from table_qa_agent.client import OpenAICompatibleVLClient
from table_qa_agent.config import ProviderConfig


def test_openai_compatible_client_parses_native_tool_call() -> None:
    captured: dict[str, object] = {}

    def create(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        function = SimpleNamespace(
            name="inspect_table_region",
            arguments='{"query":"营业收入","reason":"数字太小"}',
        )
        message = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(id="call_1", function=function)],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13),
        )

    client = object.__new__(OpenAICompatibleVLClient)
    client.config = ProviderConfig(max_retries=0, json_mode=True)
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    completion = client.complete(
        system_prompt="system",
        user_text="question",
        image_content=[],
        tools=AGENT_TOOL_DEFINITIONS,
    )

    assert completion.text == ""
    assert completion.tool_calls[0].name == "inspect_table_region"
    assert completion.tool_calls[0].arguments["query"] == "营业收入"
    assert completion.usage.total_tokens == 13
    assert captured["tool_choice"] == "auto"
    assert captured["parallel_tool_calls"] is False
    assert "response_format" not in captured
