from pathlib import Path

from table_qa_agent.config import load_config


def test_dotenv_and_base_url_alias(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("provider:\n  model: from-yaml\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "DASHSCOPE_API_KEY=test-key\n"
        "BASE_URL=https://example.test/v1\n"
        "TABLE_QA_MODEL=from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("TABLE_QA_BASE_URL", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.delenv("TABLE_QA_MODEL", raising=False)

    config = load_config(config_file)

    assert config.provider.base_url == "https://example.test/v1"
    assert config.provider.model == "from-dotenv"
    assert config.planning.model_intent_enabled is True
    assert config.planning.fallback_to_rules is True
